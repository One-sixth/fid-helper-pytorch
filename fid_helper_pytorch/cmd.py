from fid_helper_pytorch import FidHelper, SUPPORT_RESAMPLE_MODES, SUPPORT_RESAMPLE_SUFFIX, INPUT_RANGE_TF, INPUT_RANGE_PT
import click
import torch
import random
import numpy as np


# all support resample_mode
all_support_resample_mode = []
for suffix in SUPPORT_RESAMPLE_SUFFIX:
    for mode in SUPPORT_RESAMPLE_MODES:
        mode = mode + suffix
        all_support_resample_mode.append(mode)


def is_none(s):
    if s is None or str(s).lower() == 'none':
        return True
    return False


def parse_size(s):
    if is_none(s):
        return None
    else:
        h, w = s.split('x', 2)
        size = [int(h), int(w)]
        return size


def parse_range(s):
    if is_none(s):
        return None

    if str(s).lower() == 'tf':
        r = INPUT_RANGE_TF
    elif str(s).lower() == 'pt':
        r = INPUT_RANGE_PT
    else:
        low, high = s.split('-', 2)
        r = np.int32([float(low), float(high)])
    return r


def parse_device(s):
    if is_none(s):
        s = -2

    if s == -2:
        # auto select device
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
    elif s == -1:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{s}')
    return device


@click.group()
@click.option('--device',   help='Which device will used. -2=auto, -1=cpu, 0=cuda:0 1=cuda:1 etc.',    type=click.IntRange(min=-2),                 default=-2,                   show_default=True)
@click.option('--size',     help='Input image size. Allow "HxW" or "none".',                           type=str,                                    default='299x299',            show_default=True)
@click.option('--resample', help='Resample mode.',                                                     type=click.Choice(all_support_resample_mode),default='nv_bilinear_float',  show_default=True)
@click.option('--bs',       help='Batch size.',                                                        type=click.IntRange(min=1),                  default=32,                   show_default=True)
@click.option('--model',    help='Feature extracter name or path.',                                    type=str,                                    default=':default_1',         show_default=True)
@click.option('--field',    help='Feature extracter input range. Allow "tf", "pt" or "low,high".',     type=str,                                    default='tf',                 show_default=True)
@click.option('--workers',  help='How many worker to load data.',                                      type=click.IntRange(min=0),                  default=2,                    show_default=True)
@click.option('--verbose',  help='Show progressbar.',                                                  type=bool,                                   default=True,                 show_default=True)
@click.pass_context
def main(ctx, device, size, resample, bs, model, field, workers, verbose):

    # fix seed. setting deterministic and disable tf32
    # disable tf32 and benchmark is very importance
    torch.manual_seed(0)
    random.seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    ctx.ensure_object(dict)

    device = parse_device(device)

    model_input_range = parse_range(field)

    size = parse_size(size)
    assert resample in all_support_resample_mode
    ctx.obj['bs'] = bs
    ctx.obj['workers'] = workers
    ctx.obj['verbose'] = verbose

    fidhelper = FidHelper(model, model_input_range, size, resample, device)
    ctx.obj['fidhelper'] = fidhelper


@main.command()
@click.argument('img_folder1', type=str)
@click.argument('img_folder2', type=str)
@click.pass_context
def compare(ctx, img_folder1, img_folder2):
    fidhelper: FidHelper = ctx.obj['fidhelper']
    score = fidhelper.compute_fid_score_from_dir(
        img_folder1,
        img_folder2,
        batch_size=ctx.obj['bs'],
        num_workers=ctx.obj['workers'],
        verbose=ctx.obj['verbose'])
    print(f'FID: {score}')


@main.command()
@click.argument('folder',   type=str)
@click.argument('stat_out', type=str, default='stat_out.pkl')
@click.pass_context
def extract(ctx, folder, stat_out):
    fidhelper: FidHelper = ctx.obj['fidhelper']
    fidhelper.compute_eval_stat_from_dir(
        folder,
        batch_size=ctx.obj['bs'],
        num_workers=ctx.obj['workers'],
        verbose=ctx.obj['verbose'])
    fidhelper.save_eval_stat_dict(stat_out)
    print('Complete.')


if __name__ == '__main__':
    main()
