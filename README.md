# fid-helper-pytorch

## Note: python 3.9 is the minimum required version  
## 注意：python 3.9 是最低要求版本  

This is a FID helper tool.  
Provides a simple and convenient interface to calculate FID.  

The repository uses pytorch jit script to package models, including one default models.  
:default_1 with model and weights from styleganv3.

The folder make_default_model contains the build scripts for the default model.  

I wanted to implement a simple fid tool, but found clean-fid by the way and it works great.  
So by the way, the calculation method of clean-fid is added.  

They all have the same weight, but their outputs are slightly different, possibly because of jit packaging, calculation order, and random number-related issues.  

While supporting clean-fid calculations, I included as many sampling methods as possible, although it didn't help. (laugh)  

By the way, if you want to use the latest pytorch antialiasing methods torch_bilinear_aa and torch_bicubic_aa.  
You need to update to pytorch-1.11 or newer.  

这是一个FID助手工具。  
提供了一个简单方便的接口来计算FID。  

仓库使用pytorch jit script来打包模型，包含了一个默认模型。  
:default_1 的模型和权重来自 styleganv3 fid tool。

文件夹 make_default_model 包含了默认模型的编译脚本。

我本想实现一个简单的fid工具，但顺便发现了clean-fid和它很棒的效果。  
所以顺便追加了clean-fid的计算方式。  

他们均有相同的权重，但可能是因为jit包装，计算顺序，随机数相关的的问题，它们的输出均有细微的差异。  

支持clean-fid的计算方式同时，我把尽可能多的采样方法涵盖进去，尽管这并没有什么用。(笑)  

对了，如果你希望使用最新的pytorch抗锯齿的方法 torch_bilinear_aa 和 torch_bicubic_aa。  
你需要更新到 pytorch-1.11 以及更新版本。  


# Some chores / 一些杂事

Originally included 3 default models.  
:default_1 The model and weights are from styleganv3 fid tool.  
:default_2 The model and weights are from pytorch_fid.  
:default_3 The model and weights are from clean-fid.  
But I found that they are exactly the same and I ended up keeping only the first model.  
For the second and third models you can find them build script in make_default_model.  

原本包含了 3个默认模型。  
:default_1 的模型和权重来自 styleganv3 fid tool。  
:default_2 的模型和权重来自 pytorch_fid。  
:default_3 的模型和权重来自 clean-fid。 
但我发现它们是完全一样的，最后我只保留了第一个模型。  
第二个和第三个模型你可以在 make_default_model 里面找到它们的构建脚本。  

# Setup / 安装

Install from pip
从pip中安装  
```commandline
pip install fid-helper-pytorch
```
Install from source
从源码中安装  
```commandline
git clone https://github.com/One-sixth/fid-helper-pytorch
cd fid-helper-pytorch
pip install -e .
```
Build wheel
构建轮子
```commandline
git clone https://github.com/One-sixth/fid-helper-pytorch
cd fid-helper-pytorch
python setup.py bdist_wheel
# The wheel that you can found in dist folder.
```

# How to use / 如何使用

Compare two folders directly, the default FID calculation method is consistent with pytorch-fid.  
Among them, both dir1 and dir2 can be image folders, image zip archives or statistical files.  
直接比对两个文件夹，默认FID计算方式与pytorch-fid一致。  
其中，dir1 和 dir2 都可以是 图片文件夹，图片zip压缩包或统计文件。  
```commandline
# Compare folder1 and folder2 / 比较两个文件夹
fid-helper compare dir1 dir2

# Compare zip1 and zip2 / 比较两个zip压缩包
fid-helper compare dir1.zip dir2.zip

# Compare stat1 and stat2 / 比较两个统计文件
fid-helper compare dir1_stat.pkl dir2_stat.pkl

# You also can swap them freely. / 你也可以自由交换它们

# Compare folder1 and zip2 / 比较文件夹和zip压缩包
fid-helper compare dir1 dir2.zip

# Compare stat1 and folder2 / 毕竟统计文件和文件夹
fid-helper compare dir1_stat.pkl dir2.zip

```

Generate stat files for the target folder or zip archive, which is convenient for fast evaluation.  
给目标图片文件夹，图片压缩包生成统计文件，便于后续快速评估  
```commandline
# Extract stats file for folder1 / 导出文件夹1的统计文件
fid-helper extract dir1 dir1_stat.pkl

# Extract stats file for folder1 / 导出压缩包的统计文件
fid-helper extract dir1.zip dir1_stat.pkl

# Extract stats file for stats file :) / 导出统计文件的统计文件 :)
fid-helper extract dir1_stat.pkl dir1_stat_2.pkl
```

# Compare with other libraries / 与其他库进行比较

-------------------------------------------------------------------------------------
If you want to get scores consistent with clean-fid legacy_tensorflow mode  
If you want to get scores consistent with NVlabs/stylegan3’s FID tool  
如果你希望获得与 clean-fid legacy_tensorflow 模式一致的分数  
如果你希望获得与 NVlabs/stylegan3’s FID tool 一致的分数  
```commandline
fid-helper-pytorch compare folder1 folder2
```
or
```commandline
fid-helper-pytorch --model=:default_1 --resample=nv_bilinear_float --field tf compare folder1 folder2
```
-------------------------------------------------------------------------------------
If you want to get the same score as pytorch-fid.  
If you want to get scores consistent with clean-fid legacy_pytorch mode.  
如果你希望获得与 pytorch-fid 一致的分数。  
如果你希望获得与 clean-fid legacy_pytorch 模式一致的分数。
or
```commandline
fid-helper-pytorch --model=:default_1 --resample=torch_bilinear_float --field pt compare folder1 folder2
```
-------------------------------------------------------------------------------------
If you want to get scores consistent with clean-fid clean mode.
如果你希望获得与 clean-fid clean 模式一致的分数  
clean-fid  
```commandline
fid-helper-pytorch --model=:default_1 --resample=pil_bicubic_float --field tf compare folder1 folder2
```
or
```commandline
fid-helper-pytorch --model=:default_1 --resample=torch_bicubic_float --field tf compare folder1 folder2
```
-------------------------------------------------------------------------------------


# FID test / FID 测试
All test data are generated from stylegan3.  
_gen_sample_from_stylegan3/_gen_sample.py generates source code for test data.  
_gen_sample_from_stylegan3/_batch_compare_fid.py is the full FID test.  

Here is my generated test data. You can reproduce my results from this test data.  
Each archive has 10000 png images.  

全部测试数据均生成自 stylegan3。  
_gen_sample_from_stylegan3/_gen_sample.py 为测试数据生成源码。  
_gen_sample_from_stylegan3/_batch_compare_fid.py 为完整的FID测试。  

这是我生成的测试数据。你可以从该测试数据中复现我的结果。  
每个压缩包均有 10000 张png图像。  
```
https://anonfiles.com/dfDej8Oaxe/sample_1_zip
https://anonfiles.com/V3N5j6O1x3/sample_2_zip
```

## Score comparison / 分数比较

More detailed data you can find here. [compare.txt](compare-2022-03-16_18-15-31_512013.txt)  
更详细的数据你可以在这里找到 [compare.txt](compare-2022-03-16_18-15-31_512013.txt)  
```markdown
--
# pytorch-fid
3.772843757504859

# clean-fid mode=legacy_pytorch
3.772843691229639

# fid-helper-pytorch model=:default_1 input_range=pytorch resample=torch_bilinear_float
3.7728430784583953

---
# clean-fid mode=clean
3.7965344580221654

# fid-helper-pytorch model=:default_1 input_range=tf resample=pil_bicubic_float
3.79653415593325

# fid-helper-pytorch model=:default_1 input_range=tf resample=torch_bicubic_aa_float
3.7965341239671466

---
# clean-fid mode=legacy_tensorflow
3.773897315245904

# stylegan3 fid tool
none. please see clean-fid legacy_tensorflow

# fid-helper-pytorch model=:default_1 input_range=tf resample=nv_bilinear_float
3.7738974250271213

---
```


# In-app FID compute / 应用内FID计算

Example_1 Compute fid by yourself.  
```python

import numpy as np
from fid_helper_pytorch import FidHelper, INPUT_RANGE_TF


fidhelper = FidHelper(':default_1', INPUT_RANGE_TF, resample_mode='nv_bilinear_float', device='cuda:0')

# Compute real image stat by yourself.
fidhelper.begin_ref_stat()
for _ in range(16):
    batch_image = np.random.randint(0, 255, size=[6, 3, 256, 256], dtype=np.uint8)
    fidhelper.update_ref_stat(batch_image, data_range=(0, 255))
    
fidhelper.finish_ref_stat()

# Compute fake image stat by yourself.
fidhelper.begin_eval_stat()
for _ in range(16):
    gen_images = np.random.uniform(-1., 1., size=[18, 3, 256, 256])
    fidhelper.update_eval_stat(gen_images, data_range=[-1, 1])
fidhelper.finish_eval_stat()

# Compute fid.
fid_score = fidhelper.compute_fid_score()
print('FID:', fid_score)

```

Example_2 Compute stat and fid from folder and cycle.  

```python
import torch
from fid_helper_pytorch import FidHelper

fidhelper = FidHelper(':default_1')
batch_size = 4
num_workers = 1

real_img_dir = 'real_img'

# Compute real image stat from image folder.
fidhelper.compute_ref_stat_from_dir(real_img_dir, batch_size=batch_size, num_workers=num_workers)

# Compute fake image stat by yourself.
fidhelper.begin_eval_stat()
for _ in range(16):
    gen_images = torch.rand(size=[18, 3, 256, 256])
    fidhelper.update_eval_stat(gen_images, data_range=[0, 1])
fidhelper.finish_eval_stat()

# Compute fid.
fid_score = fidhelper.compute_fid_score()
print('FID:', fid_score)

```

Example_3 Compute fid from zip and get last fid.  

```python
from fid_helper_pytorch import FidHelper

fidhelper = FidHelper(':default_1')
batch_size = 4
num_workers = 1

real_img_arc = 'real_img.zip'
fake_img_arc = 'fake_img.zip'

# Compute real image stat from image folder.
fid_score = fidhelper.compute_fid_score_from_dir(real_img_arc, fake_img_arc, batch_size=batch_size, num_workers=num_workers, verbose=True)
print('FID:', fid_score)

# Compute fid.
fid_score = fidhelper.fid_score
print('last FID:', fid_score)

```

Example_4 Compute stat and fid from folder.  
```python
from fid_helper_pytorch import FidHelper

fidhelper = FidHelper(resample_mode='pil_bicubic_float')

fake_img_dir = 'fake_img'
real_img_dir = 'real_img'

# Compute real image stat from image folder.
fidhelper.compute_ref_stat_from_dir(fake_img_dir, batch_size=16, num_workers=2, verbose=True)

# Compute fake image stat from image folder.
fidhelper.compute_eval_stat_from_dir(real_img_dir, batch_size=24, num_workers=3, verbose=True)

# Compute fid.
fid_score = fidhelper.compute_fid_score()
print('FID:', fid_score)

```

Example_5 Compute stat file and save to disk.  
```python
from fid_helper_pytorch import FidHelper

fidhelper = FidHelper(':default_1')

real_img_dir = 'real_img'
fake_img_dir = 'fake_img'

# Compute real image stat from image folder.
fidhelper.compute_ref_stat_from_dir(real_img_dir)

# Save stat file to disk.
fidhelper.save_ref_stat_dict('out_real_stat.pkl')

# Compute fake image stat from image folder.
fidhelper.compute_eval_stat_from_dir(fake_img_dir)

# Save stat file to disk.
fidhelper.save_eval_stat_dict('out_fake_stat.pkl')

```

Example_6 Load stat file and compute fid.  
```python
from fid_helper_pytorch import FidHelper

fidhelper = FidHelper(':default_1')

# Get fake image stat from stat file.
fidhelper.load_ref_stat_dict('fake_stat.pkl')

# Get real image stat from stat file.
fidhelper.load_eval_stat_dict('real_stat.pkl')

fid_score = fidhelper.compute_fid_score()
print('FID:', fid_score)

```

Example_7 Change paramters anywhere.  

```python
from fid_helper_pytorch import FidHelper

fidhelper = FidHelper(':default_1')

# Change model anywhere.
fidhelper.load_model('xxx.pt', model_input_range=None)

# Change device anywhere.
fidhelper.change_device('cuda:1')

# Change model input range anywhere.
fidhelper.change_model_input_range([0, 1])

# Change resample_mode anywhere.
fidhelper.change_resample_mode('torch_bilinear_float')

# Change resize_hw anywhere.
fidhelper.change_resize_hw([320, 320])

```


# Reference / 参考引用

Thank you very much for the open source repository of the big guys. Without their selfless dedication, this repository would not exist.  
If you like this repo, please star the repo of the following big guys.  

非常感谢大佬们的开源仓库，没有他们无私奉献，这个库将不可能存在。  
如果你喜欢本仓库，请为下面大佬的仓库star。  

https://github.com/mseitzer/pytorch-fid The most popular pytorch-fid repository / 最流行的pytorch-fid仓库  
https://github.com/GaParmar/clean-fid   The latest clean-fid repository / 最新的clean-fid仓库  
https://github.com/NVlabs/stylegan3     The official repository of stylegan3 / stylegan3的官方仓库  
