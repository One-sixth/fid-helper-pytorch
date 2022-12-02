# _gen_sample_from_stylegan3

The source code is from https://github.com/NVlabs/stylegan3 .  
The model wegiht is from https://catalog.ngc.nvidia.com/orgs/nvidia/teams/research/models/stylegan3/files .  

~~And I patch the custom_ops.py L30 to support VS2022 x64 complier.~~  
I have been make styleganv3 generator never need to compile.  

stylegan3-t-ffhqu-256x256-shim.pkl was extracted from stylegan3-t-ffhqu-256x256.pkl .  
It only has G_ema net.  

# How to use?  
~~Just install VS2022, cuda 11.x and cudnn.~~  
~~And then run _gen_sample.py .~~  

Just run python _gen_sample.py .  
The generated images can found in sample_1 and sample_2 folder.  
