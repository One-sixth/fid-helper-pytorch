import numpy as np
import torch
import torch.nn.functional as F
import PIL.Image
import cv2

# Use tuples to avoid unexpected modifications.
SUPPORT_RESAMPLE_MODES = ('nv_bilinear',
                          'torch_bilinear', 'torch_bilinear_aa', 'torch_bicubic', 'torch_bicubic_aa', 'torch_area', 'torch_nearest', 'torch_nearest_exact',
                          'cv_bilinear', 'cv_bilinear_exact', 'cv_nearest', 'cv_nearest_exact', 'cv_bits', 'cv_bits2', 'cv_area', 'cv_lanczos',
                          'pil_bilinear', 'pil_bicubic', 'pil_nearest', 'pil_box', 'pil_lanczos', 'pil_hamming')

# Use tuples to avoid unexpected modifications.
SUPPORT_RESAMPLE_SUFFIX = ('', '_float')

_error_msg_bad_mode = f'Error! Bad resample_mode {{ori_mode}}. The valid resample_mode has {str(SUPPORT_RESAMPLE_MODES)} and their float version.'


def check_valid_resample_mode(mode: str):
    for suffix in SUPPORT_RESAMPLE_SUFFIX:
        m = mode.removesuffix(suffix)
        if m in SUPPORT_RESAMPLE_MODES:
            return
    raise AssertionError(_error_msg_bad_mode.format(mode))


# Use jit to speed up.
@torch.jit.script
def nv_bilinear(x, new_hw: tuple[int, int]=(299, 299)):
    batch_size, channels, height, width = x.shape
    new_height, new_width = new_hw
    theta = torch.eye(2, 3, device=x.device)
    theta[0, 2] += theta[0, 0] / width - theta[0, 0] / new_width
    theta[1, 2] += theta[1, 1] / height - theta[1, 1] / new_height
    theta = theta.to(x.dtype).unsqueeze(0).repeat([batch_size, 1, 1])
    grid = torch.nn.functional.affine_grid(theta, [batch_size, channels, new_height, new_width], align_corners=False)
    x = torch.nn.functional.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=False)
    return x


def torch_resample(x, new_hw, mode, is_tensor=False):
    assert isinstance(x, (np.ndarray, torch.Tensor))
    float_mode = mode.endswith('_float')
    ori_mode = mode

    if is_tensor:
        # fast channel
        # tensor mode will always float32
        # bchw
        x = x.type(torch.float32)
    else:
        # hwc->bhwc->bchw
        x = torch.from_numpy(x)
        x = x.type(torch.float32)[None,].permute(0, 3, 1, 2)

    if float_mode:
        mode = mode.removesuffix('_float')

    if mode == 'nv_bilinear':
        x = nv_bilinear(x, new_hw)
    elif mode == 'torch_bilinear':
        x = F.interpolate(x, new_hw, mode='bilinear', align_corners=False)
    elif mode == 'torch_bilinear_aa':
        x = F.interpolate(x, new_hw, mode='bilinear', align_corners=False, antialias=True)
    elif mode == 'torch_bicubic':
        x = F.interpolate(x, new_hw, mode='bicubic', align_corners=False)
    elif mode == 'torch_bicubic_aa':
        x = F.interpolate(x, new_hw, mode='bicubic', align_corners=False, antialias=True)
    elif mode == 'torch_area':
        x = F.interpolate(x, new_hw, mode='area')
    elif mode == 'torch_nearest':
        x = F.interpolate(x, new_hw, mode='nearest')
    elif mode == 'torch_nearest_exact':
        x = F.interpolate(x, new_hw, mode='nearest-exact')
    else:
        raise AssertionError(_error_msg_bad_mode.format(ori_mode))

    if not float_mode:
        x = x.round_().clamp_(0, 255)

    if not is_tensor:
        # bchw->bhwc->hwc
        x = x.permute(0, 2, 3, 1)[0]
        if not float_mode:
            x = x.type(torch.uint8)
        x = x.numpy()

    return x


def cv_resample(x, new_hw, mode: str):
    assert isinstance(x, np.ndarray)
    ori_mode = mode
    float_mode = mode.endswith('_float')

    if float_mode:
        mode = mode.removesuffix('_float')
        x = np.asarray(x, np.float32)
    else:
        x = np.asarray(x, np.uint8)

    if mode == 'cv_bilinear':
        x = cv2.resize(x, tuple(new_hw[::-1]), interpolation=cv2.INTER_LINEAR)
    elif mode == 'cv_bilinear_exact':
        x = cv2.resize(x, tuple(new_hw[::-1]), interpolation=cv2.INTER_LINEAR_EXACT)
    elif mode == 'cv_bicubic':
        x = cv2.resize(x, tuple(new_hw[::-1]), interpolation=cv2.INTER_CUBIC)
    elif mode == 'cv_nearest':
        x = cv2.resize(x, tuple(new_hw[::-1]), interpolation=cv2.INTER_NEAREST)
    elif mode == 'cv_nearest_exact':
        x = cv2.resize(x, tuple(new_hw[::-1]), interpolation=cv2.INTER_NEAREST_EXACT)
    elif mode == 'cv_bits':
        x = cv2.resize(x, tuple(new_hw[::-1]), interpolation=cv2.INTER_BITS)
    elif mode == 'cv_bits2':
        x = cv2.resize(x, tuple(new_hw[::-1]), interpolation=cv2.INTER_BITS2)
    elif mode == 'cv_area':
        x = cv2.resize(x, tuple(new_hw[::-1]), interpolation=cv2.INTER_AREA)
    elif mode == 'cv_lanczos':
        x = cv2.resize(x, tuple(new_hw[::-1]), interpolation=cv2.INTER_LANCZOS4)

    else:
        raise AssertionError(_error_msg_bad_mode.format(ori_mode))

    return x


def pil_resample(x, new_hw, mode):
    # 要求输入范围是 [0, 255]，要求输入是numpy数组，要求输入类型是uint8或者float32
    assert isinstance(x, np.ndarray)

    if mode.endswith('_float'):
        x = np.float32(x)
        x = [PIL.Image.fromarray(x[..., i], mode='F') for i in range(x.shape[-1])]
    else:
        x = x.astype(np.uint8)
        x = PIL.Image.fromarray(x, mode='RGB')

    if mode == 'pil_bilinear':
        x = x.resize(new_hw[::-1], resample=PIL.Image.BILINEAR)
    elif mode == 'pil_bicubic':
        x = x.resize(new_hw[::-1], resample=PIL.Image.BICUBIC)
    elif mode == 'pil_nearest':
        x = x.resize(new_hw[::-1], resample=PIL.Image.NEAREST)
    elif mode == 'pil_box':
        x = x.resize(new_hw[::-1], resample=PIL.Image.BOX)
    elif mode == 'pil_lanczos':
        x = x.resize(new_hw[::-1], resample=PIL.Image.LANCZOS)
    elif mode == 'pil_hamming':
        x = x.resize(new_hw[::-1], resample=PIL.Image.HAMMING)

    elif mode == 'pil_bilinear_float':
        x = [xi.resize(new_hw[::-1], resample=PIL.Image.BILINEAR) for xi in x]
    elif mode == 'pil_bicubic_float':
        x = [xi.resize(new_hw[::-1], resample=PIL.Image.BICUBIC) for xi in x]
    elif mode == 'pil_nearest_float':
        x = [xi.resize(new_hw[::-1], resample=PIL.Image.NEAREST) for xi in x]
    elif mode == 'pil_box_float':
        x = [xi.resize(new_hw[::-1], resample=PIL.Image.BOX) for xi in x]
    elif mode == 'pil_lanczos_float':
        x = [xi.resize(new_hw[::-1], resample=PIL.Image.LANCZOS) for xi in x]
    elif mode == 'pil_hamming_float':
        x = [xi.resize(new_hw[::-1], resample=PIL.Image.HAMMING) for xi in x]

    else:
        raise AssertionError(_error_msg_bad_mode.format(mode))

    if isinstance(x, list):
        x = [np.asarray(xi) for xi in x]
        x = np.stack(x, 2)
    else:
        x = np.asarray(x)

    return x


def resample_func(x, new_hw, mode):
    check_valid_resample_mode(mode)
    if mode.startswith(('nv_', 'torch_')):
        x = torch_resample(x, new_hw, mode)
    elif mode.startswith('cv_'):
        x = cv_resample(x, new_hw, mode)
    elif mode.startswith('pil_'):
        x = pil_resample(x, new_hw, mode)
    else:
        raise AssertionError(_error_msg_bad_mode.format(mode))
    return x


def tensor_resample_func(x, new_hw, mode, quant=True):
    '''
    :param x:       Input tensor
    :param new_hw:  New image size_hw
    :param mode:    Resample mode.
    :param quant:   Whether to quantize the input.
    :return:
    '''
    check_valid_resample_mode(mode)
    assert isinstance(x, torch.Tensor)
    assert new_hw is not None
    assert x.dtype == torch.float32
    # x must be in range [0, 1] and type is float32. Because x has been clamp.
    if mode.startswith(('nv_', 'torch_')):
        # fast channel
        # force quat.
        x = x.mul(255)
        if quant:
            x = x.round_() # .type(torch.uint8) # convert to uint8 is not necessary
        x = torch_resample(x, new_hw, mode, is_tensor=True)
        x = x.div(255)
    else:
        # slow channel
        # rescale value range from [0, 1] float32 to [0, 255] uint8
        # convert x[bchw] -> [x1[hwc], x2[hwc], ...]
        x_device = x.device
        x_dtype = x.dtype
        # force quat.
        x = x.mul(255).permute(0, 2, 3, 1)
        if quant:
            x = x.round_() # .type(torch.uint8) # convert to uint8 is not necessary
        # need resample each by each.
        x = list(x.cpu().numpy())
        if mode.startswith('cv_'):
            x = [cv_resample(xi, new_hw, mode) for xi in x]
        elif mode.startswith('pil_'):
            x = [pil_resample(xi, new_hw, mode) for xi in x]
        else:
            raise AssertionError(f'Error! Bad resample_mode {mode}.')
        # x.dtype must be is np.uint8 or np.float32
        x = np.stack(x, 0)
        assert x.dtype in (np.uint8, np.float32)
        # rescale value range [0, 255] to [0, 1]
        x = torch.from_numpy(x).to(device=x_device, dtype=x_dtype).div_(255).permute(0, 3, 1, 2)
    return x
