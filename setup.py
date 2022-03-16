from setuptools import setup  # , find_packages
import sys
import os
import importlib
sys.path.insert(0, os.path.dirname(__file__) + '/fid_helper_pytorch')
ver = importlib.import_module('__version__')
ver_str = ver.__version__


setup(
    name='fid-helper-pytorch',
    version=ver_str,
    description='An easy-to-use pytorch fid toolkit.',
    author='onesixth',
    author_email='noexist@noexist.noexist',
    url='https://github.com/One-sixth/fid-helper-pytorch',
    install_requires=['requests', 'click', 'numpy', 'tqdm', 'pillow', 'scipy', 'filetype', 'opencv-python', 'loguru', 'torch>=1.10'],
    entry_points={'console_scripts': ['fid-helper-pytorch = fid_helper_pytorch.cmd:main']},
    packages=['fid_helper_pytorch'],
    package_data={'fid_helper_pytorch': ['default_1.pt']},
)
