import os
import sys
import random
from loguru import logger
from pytorch_fid.fid_score import calculate_fid_given_paths as pytorch_fid_calculate_fid_given_paths
from fid_helper_pytorch import FidHelper, SUPPORT_RESAMPLE_MODES, SUPPORT_RESAMPLE_SUFFIX, INPUT_RANGE_TF, INPUT_RANGE_PT
from cleanfid.fid import compute_fid as clean_fid_compute_fid
import torch


if __name__ == '__main__':

    # --------------------------------------------------------------------------------------------------------------

    # fix seed. setting deterministic and disable tf32
    # disable tf32 and benchmark is very importance
    assert torch.cuda.is_available()
    torch.manual_seed(0)
    random.seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # --------------------------------------------------------------------------------------------------------------

    logger.add('compare-{time}.txt')
    logger.info(f'system platform: {sys.platform}')
    logger.info(f'pytorch version: {torch.version.__version__}')
    logger.info(f'pytorch cuda version: {torch.version.cuda}')
    logger.info(f'pytorch git_version: {torch.version.git_version}')

    # --------------------------------------------------------------------------------------------------------------

    cur_dir = os.path.dirname(__file__)

    sample_1_dir = cur_dir + '/sample_1'
    sample_2_dir = cur_dir + '/sample_2'
    # sample_1_dir = cur_dir + '/sample_1_small'
    # sample_2_dir = cur_dir + '/sample_2_small'
    device = 0
    # batch_size = 128
    batch_size = 16
    if sys.platform == 'win32':
        print('Because windows system can not fork, so windows system cannot open too many loaders.')
        n_worker = 0
    else:
        n_worker = 32

    # --------------------------------------------------------------------------------------------------------------

    logger.info(f'Used device: {torch.cuda.get_device_name(device)}')
    logger.info(f'Used sample folder 1: {sample_1_dir}')
    logger.info(f'Used sample folder 2: {sample_2_dir}')

    # --------------------------------------------------------------------------------------------------------------
    torch.cuda.empty_cache()

    logger.info('1. pytorch-fid')
    score = pytorch_fid_calculate_fid_given_paths([sample_1_dir, sample_2_dir], batch_size=batch_size, device=device, dims=2048, num_workers=n_worker)
    head_line = ''
    logger.info(f'{head_line:<40} score: {score}')

    logger.info(f'\n\n')

    # --------------------------------------------------------------------------------------------------------------
    torch.cuda.empty_cache()

    logger.info('2. clean-fid')

    old_n_worker = n_worker
    if sys.platform == 'win32':
        n_worker = 0

    for mode in ['clean', 'legacy_pytorch', 'legacy_tensorflow']:
        score = clean_fid_compute_fid(sample_1_dir, sample_2_dir, mode=mode, batch_size=batch_size, device=device, num_workers=n_worker)
        logger.info(f'{mode:<40} score: {score}')

    if sys.platform == 'win32':
        n_worker = old_n_worker

    logger.info(f'\n\n')

    # --------------------------------------------------------------------------------------------------------------
    torch.cuda.empty_cache()

    logger.info('3. fid-helper-pytorch')

    # for model_name in [':default_1', ':default_2', ':default_3']:
    # for model_name in [':default_1', ':default_2']:
    for model_name in [':default_1']:
        for input_range_name in ['tf', 'pytorch']:

            if input_range_name == 'tf':
                input_range = INPUT_RANGE_TF
            elif input_range_name == 'pytorch':
                input_range = INPUT_RANGE_PT
            else:
                raise AssertionError('Error! Invalid input_range.')

            for mode_suffix in SUPPORT_RESAMPLE_SUFFIX:
                for mode in SUPPORT_RESAMPLE_MODES:
                    mode = mode+mode_suffix

                    fidhelper = FidHelper(model_name, input_range, resample_mode=mode, device=f'cuda:{device}')

                    score = fidhelper.compute_fid_score_from_dir(sample_1_dir, sample_2_dir, batch_size=batch_size, num_workers=n_worker, verbose=True)

                    head_line = f'{input_range_name} {model_name} {mode}'
                    logger.info(f'{head_line:<40} score: {score}')

                    del fidhelper
                    torch.cuda.empty_cache()

    logger.info(f'\n\n')

    logger.info('Complete.')
