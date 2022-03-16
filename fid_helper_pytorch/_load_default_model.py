import os
import sys

try:
    from ._download_tool import download_file, sha1_check
except (ModuleNotFoundError, ImportError):
    from _download_tool import download_file, sha1_check


model_links = {
    ':default_1': 'https://github.com/One-sixth/fid-helper-pytorch/raw/main/fid_helper_pytorch/default_1.pt',
}

model_sha1_code = {
    ':default_1': 'A16E4D89C0ECBC17A915DD76EA9E41211AA82562',
}


def get_default_model_dir():
    '''
    Used to persist models to avoid repeated downloads.
    :return:
    '''
    default_model_dir = os.path.dirname(__file__)

    # if sys.platform == 'win32':
    #     default_model_dir = os.environ['USERPROFILE'] + '/.cache/fid-helper-pytorch'
    # elif sys.platform == 'linux':
    #     default_model_dir = os.environ['HOME'] + '/.cache/fid-helper-pytorch'

    return default_model_dir


def load_default_model(name):
    '''
    Eventually I found out that the models were identical, and got drunk. . .
    :param name:
    :return:
    '''

    default_model_dir = get_default_model_dir()
    os.makedirs(default_model_dir, exist_ok=True)

    if name == ':default_1':
        feature_extractor_path = default_model_dir+'/default_1.pt'
        # feature_extractor_path, feature_extractor_input_range = default_model_dir+'/default_1.pt', (-1, 0.9921875)
    # elif name == ':default_2':
        # feature_extractor_path, feature_extractor_input_range = default_model_dir+'/default_1.pt', (-1, 1)
    #     feature_extractor_path, feature_extractor_input_range = default_model_dir+'/default_2.pt', (-1, 1)
    # elif name == ':default_3':
    #     feature_extractor_path, feature_extractor_input_range = default_model_dir+'/default_3.pt', (-1, 0.9921875)
    else:
        raise AssertionError('Error! Unknow default model name.')

    if not os.path.isfile(feature_extractor_path) or not sha1_check(feature_extractor_path, model_sha1_code[name]):
        link = model_links[name]
        print(f'Info! Downloading {name} model. URL: {link} FILE: {feature_extractor_path}')
        download_file(link, feature_extractor_path, model_sha1_code[name])

    # return feature_extractor_path, feature_extractor_input_range
    return feature_extractor_path
