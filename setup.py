from setuptools import setup  # , find_packages

setup(
    name='fid-helper-pytorch',
    version='0.1.0',
    description='An easy-to-use pytorch fid toolkit.',
    author='onesixth',
    author_email='noexist@noexist.noexist',
    url='https://github.com/One-sixth/fid-helper-pytorch',
    install_requires=['requests', 'click', 'numpy', 'tqdm', 'pillow', 'scipy', 'filetype', 'opencv-python', 'loguru', 'torch>=1.10'],
    entry_points={'console_scripts': ['fid-helper-pytorch = fid_helper_pytorch.cmd:main']},
    packages=['fid_helper_pytorch'],
    package_data={'fid_helper_pytorch': ['default_1.pt']},
)
