import glob
import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import Union
import PIL.Image
import cv2
from io import IOBase
import scipy.linalg
import zipfile
import warnings
import filetype
import typing
# for support relative import
try:
    from ._load_default_model import load_default_model
    from ._resample_func import SUPPORT_RESAMPLE_MODES, SUPPORT_RESAMPLE_SUFFIX, check_valid_resample_mode, resample_func, tensor_resample_func
except (ImportError, ModuleNotFoundError):
    from _load_default_model import load_default_model
    from _resample_func import SUPPORT_RESAMPLE_MODES, SUPPORT_RESAMPLE_SUFFIX, check_valid_resample_mode, resample_func, tensor_resample_func

# -------------------------------------------------------------------------------------------------------------

INPUT_RANGE_PT = (-1, 1)
INPUT_RANGE_TF = (-1, 0.9921875)

# -------------------------------------------------------------------------------------------------------------

# Use list. Easy to add new image ext.
SUPPORT_IMG_EXT = ['.jpg', '.jpeg', '.tif', '.tiff', '.png', '.bmp', '.webp']


class FolderImageDataset(Dataset):
    def __init__(self, dir_path, resize_hw: Union[None, tuple]=None, resample_mode=None):
        '''
        :param dir_path: allow ImageDirPath or ImageZipPath. It can automatically recurse folders to find images.
        :param resize_hw: if none. will not resize.
        '''
        super().__init__()
        self.arc = None

        if os.path.isfile(dir_path):
            self.arc = zipfile.ZipFile(dir_path)
            files = self.arc.namelist()
        elif os.path.isdir(dir_path):
            files = glob.glob(f'{dir_path}/**/*', recursive=True)
        else:
            raise AssertionError(f'Error! The input "{dir_path}" is not folder or zipfile.')

        self.dir_path = dir_path

        img_paths = []
        for fp in files:
            if fp.endswith(tuple(SUPPORT_IMG_EXT)):
                img_paths.append(fp)

        assert len(img_paths) > 0, f'Error! The input "{dir_path}" can not found any image. The valid image extname has {str(SUPPORT_IMG_EXT)}.'

        self.img_paths = img_paths
        self.resize_hw = resize_hw
        self.resample_mode = resample_mode

        # used for windows multiprocess
        self._offload()

    def __len__(self):
        return len(self.img_paths)

    def _offload(self):
        # used for windows multiprocess
        if self.arc is not None:
            self.arc = None

    def _onload(self):
        # used for windows multiprocess
        if self.arc is None and os.path.isfile(self.dir_path):
            self.arc = zipfile.ZipFile(self.dir_path)

    def __getitem__(self, item):
        self._onload()

        fp = self.img_paths[item]
        if self.arc is None:
            f = open(fp, 'rb')
        else:
            f = self.arc.open(fp, 'r')

        im = np.asarray(PIL.Image.open(f).convert('RGB'), np.uint8)
        if self.resize_hw is not None and self.resample_mode is not None and (self.resize_hw[0] != im.shape[0] or self.resize_hw[1] != im.shape[1]):
            im = resample_func(im, self.resize_hw, self.resample_mode)

        im = torch.from_numpy(im)
        # im.shape == (H, W, 3)
        return im


# -------------------------------------------------------------------------------------------------------------


class FidHelper:
    def __init__(self, model_path=':default_1', model_input_range=INPUT_RANGE_TF, resize_hw=(299, 299), resample_mode='nv_bilinear_float', device='cpu'):
        # 特征导出模型
        self.model: torch.nn.Module = None
        # 特征导出模型输入值域
        self.model_input_range = None
        self.change_model_input_range(model_input_range)
        # 特征导出模型输入尺寸
        self.resize_hw = None
        self.change_resize_hw(resize_hw)
        # 特征导出模型输入尺寸不一致时重采样方法
        self.resample_mode = None
        self.change_resample_mode(resample_mode)
        # 提取特征模型使用的设备
        self.device = None
        self.change_device(device)

        # 评估统计字典，用于记录生成图像的统计信息
        self.eval_stat_dict = None
        # 引用统计字典，用于记录真实图像的统计信息
        self.ref_stat_dict = None
        # 用于记录特征
        self.eval_feats_list = []
        self.ref_feats_list = []
        # 用于记录最后一次获得的FID分数
        self.fid_score = None
        #
        self._is_begin_eval_stat = False
        self._is_begin_ref_stat = False

        # init
        self.load_model(model_path, model_input_range)

    # -------------------------------------------------------------------------------------------------------------

    def load_model(self, model_path=':default_1', model_input_range=None, _extra_files=None, *, force_eval=False):
        assert len(model_input_range) >= 2 and model_input_range[0] < model_input_range[1]

        # Try load default model.
        if model_path.startswith(':'):
            model_path = load_default_model(model_path)

        # Direct use when model_path is a callable module.
        if isinstance(model_path, (torch.nn.Module, torch.jit.ScriptModule)):
            net = model_path.to(self.device)
            if net.training:
                if force_eval:
                    net.eval()
                else:
                    warnings.warn('Warning! Found the model is not in eval mode. This can cause some unexpected behavior.')

        # Load model from a pytorch jit serialized file.
        else:
            net = torch.jit.load(model_path, map_location='cpu', _extra_files=_extra_files).to(self.device)
        self.model = net

        if model_input_range is not None:
            self.change_model_input_range(model_input_range)

    def change_device(self, device):
        device = torch.device(device)
        self.device = device
        if self.model is not None:
            self.model.to(device)
        return self

    def change_resample_mode(self, resample_mode):
        check_valid_resample_mode(resample_mode)
        self.resample_mode = resample_mode

    def change_model_input_range(self, model_input_range):
        assert len(model_input_range) == 2 and model_input_range[0] < model_input_range[1]
        self.model_input_range = model_input_range

    def change_resize_hw(self, resize_hw):
        assert len(resize_hw) == 2 and resize_hw[0] >= 1 and resize_hw[1] >= 1
        self.resize_hw = (int(resize_hw[0]), int(resize_hw[1]))

    # -------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _load_stat_dict(file_or_dict):
        if isinstance(file_or_dict, dict):
            stat_dict = file_or_dict
        elif isinstance(file_or_dict, str):
            stat_dict = pickle.load(open(file_or_dict, 'rb'))
        elif isinstance(file_or_dict, IOBase):
            stat_dict = pickle.load(file_or_dict)
        else:
            raise AssertionError(f'Error! Invalid input param "file_or_dict". {file_or_dict}')
        return stat_dict

    def load_eval_stat_dict(self, file_or_dict):
        self.eval_stat_dict = self._load_stat_dict(file_or_dict)

    def load_ref_stat_dict(self, file_or_dict):
        self.ref_stat_dict = self._load_stat_dict(file_or_dict)

    # -------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _save_stat_dict(stat_dict, fp=None):
        if fp is not None:
            if isinstance(fp, str):
                pickle.dump(stat_dict, open(fp, 'wb'))
            elif isinstance(fp, IOBase):
                pickle.dump(stat_dict, fp)
            else:
                raise AssertionError(f'Error! Invalid input param "fp". {fp}')
        return stat_dict

    def save_eval_stat_dict(self, fp=None):
        return self._save_stat_dict(self.eval_stat_dict, fp)

    def save_ref_stat_dict(self, fp=None):
        return self._save_stat_dict(self.ref_stat_dict, fp)

    # -------------------------------------------------------------------------------------------------------------

    @torch.inference_mode()
    def extract_feats(self, batch_image, data_range, resize_hw=None, resample_mode=None, quant=False):
        if isinstance(batch_image, np.ndarray):
            batch_image = torch.from_numpy(batch_image)

        assert isinstance(batch_image, torch.Tensor)
        assert batch_image.ndim == 4 and batch_image.shape[1] == 3, 'Error! Required input tensor is NCHW and num_channel is 3.'
        assert data_range[1] > data_range[0]

        t_x = batch_image.contiguous().type(torch.float32).to(self.device)

        # 缩放输入到[0, 1]区间，并截断
        t_x = (t_x - data_range[0]) / (data_range[1] - data_range[0])
        t_x = t_x.clamp(0, 1)

        if resize_hw is not None and resample_mode is not None and (t_x.shape[2] != resize_hw[0] or t_x.shape[3] != resize_hw[1]):
            t_x = tensor_resample_func(t_x, resize_hw, resample_mode, quant)

        if self.model_input_range is not None:
            # 如果定义了特征导出器的输入域，则缩放到特征导出器的输入域，并截断
            t_x = t_x * (self.model_input_range[1] - self.model_input_range[0]) + self.model_input_range[0]
            t_x = t_x.clamp(self.model_input_range[0], self.model_input_range[1])

        feats = self.model(t_x)

        if feats.ndim > 2:
            dim = list(range(2, feats.ndim))
            feats = feats.mean(dim=dim)
        assert feats.ndim == 2
        feats = feats.cpu().numpy()
        return feats

    # -------------------------------------------------------------------------------------------------------------

    def begin_ref_stat(self):
        assert not self._is_begin_ref_stat
        self._is_begin_ref_stat = True
        self.ref_feats_list = []

    def update_ref_stat(self, batch_image, data_range, quant=True):
        '''
        :param batch_image: shape[N, C, H, W]
        :return:
        '''
        assert self._is_begin_ref_stat
        feats = self.extract_feats(batch_image, data_range=data_range, resize_hw=self.resize_hw, resample_mode=self.resample_mode, quant=quant)
        self.ref_feats_list.extend(feats)

    def finish_ref_stat(self):
        assert self._is_begin_ref_stat
        self._is_begin_ref_stat = False

        stat_dict = dict(mu=np.mean(self.ref_feats_list, axis=0), sigma=np.cov(self.ref_feats_list, rowvar=False))
        self.ref_stat_dict = stat_dict
        return stat_dict

    # -------------------------------------------------------------------------------------------------------------

    def begin_eval_stat(self):
        assert not self._is_begin_eval_stat
        self._is_begin_eval_stat = True
        self.eval_feats_list = []

    def update_eval_stat(self, batch_image, data_range, quant=True):
        '''
        :param batch_image: shape[N, C, H, W]
        :return:
        '''
        assert self._is_begin_eval_stat
        feats = self.extract_feats(batch_image, data_range=data_range, resize_hw=self.resize_hw, resample_mode=self.resample_mode, quant=quant)
        self.eval_feats_list.extend(feats)

    def finish_eval_stat(self):
        assert self._is_begin_eval_stat
        self._is_begin_eval_stat = False

        stat_dict = dict(mu=np.mean(self.eval_feats_list, axis=0), sigma=np.cov(self.eval_feats_list, rowvar=False))
        self.eval_stat_dict = stat_dict
        return stat_dict

    # -------------------------------------------------------------------------------------------------------------

    def compute_fid_score(self):
        mu1 = self.ref_stat_dict['mu']
        sigma1 = self.ref_stat_dict['sigma']

        mu2 = self.eval_stat_dict['mu']
        sigma2 = self.eval_stat_dict['sigma']

        assert mu1 is not None and sigma1 is not None, 'Error! The ref stat is empty.'
        assert mu2 is not None and sigma2 is not None, 'Error! The eval stat is empty.'

        m = np.square(mu2 - mu1).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma2, sigma1), disp=False)
        fid = np.real(m + np.trace(sigma2 + sigma1 - s * 2))
        self.fid_score = fid
        return fid

    # -------------------------------------------------------------------------------------------------------------

    def compute_ref_stat_from_dir(self, dir_path, batch_size=1, num_workers=0, verbose=False):
        if os.path.isfile(dir_path) and not filetype.guess_extension(dir_path) == 'zip':
            self.load_ref_stat_dict(dir_path)
        else:
            ds = FolderImageDataset(dir_path, self.resize_hw, self.resample_mode)
            dl = DataLoader(ds, batch_size, num_workers=num_workers)

            self.begin_ref_stat()

            for batch_im in tqdm(dl, disable=not verbose, desc='compute ref stat.'):
                assert batch_im.shape[1] == self.resize_hw[0] and batch_im.shape[2] == self.resize_hw[1]
                batch_im = batch_im.permute(0, 3, 1, 2)
                self.update_ref_stat(batch_im, data_range=[0, 255], quant=False)

            self.finish_ref_stat()

        return self.ref_stat_dict

    def compute_eval_stat_from_dir(self, dir_path, batch_size=1, num_workers=0, verbose=False):
        if os.path.isfile(dir_path) and not filetype.guess_extension(dir_path) == 'zip':
            self.load_eval_stat_dict(dir_path)
        else:
            ds = FolderImageDataset(dir_path, self.resize_hw, self.resample_mode)
            dl = DataLoader(ds, batch_size, num_workers=num_workers)

            self.begin_eval_stat()

            for batch_im in tqdm(dl, disable=not verbose, desc='compute eval stat.'):
                assert batch_im.shape[1] == self.resize_hw[0] and batch_im.shape[2] == self.resize_hw[1]
                batch_im = batch_im.permute(0, 3, 1, 2)
                self.update_eval_stat(batch_im, data_range=[0, 255], quant=False)

            self.finish_eval_stat()

        return self.eval_stat_dict

    # -------------------------------------------------------------------------------------------------------------

    def compute_fid_score_from_dir(self, ref_dir, eval_dir, batch_size=1, num_workers=0, verbose=False):
        self.compute_ref_stat_from_dir(ref_dir, batch_size=batch_size, num_workers=num_workers, verbose=verbose)
        self.compute_eval_stat_from_dir(eval_dir, batch_size=batch_size, num_workers=num_workers, verbose=verbose)
        self.compute_fid_score()
        return self.fid_score

    # -------------------------------------------------------------------------------------------------------------
