'''
原始路径，该模型为Jit模型
https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class WrapNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.jit.load('inception-2015-12-05.pt')
        self.layers = self.base.layers
        self.resize_inside = False

        # self.base.requires_grad_(False)
        self.base.eval()
        self.base.to('cpu')

    def forward(self, x):
        y = self.layers.forward(x,).view((x.shape[0], 2048))
        return y
