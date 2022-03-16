import torch
import torch.nn as nn
from inceptionv3 import InceptionV3


def patch_forward(self, img):
    '''
    略加修改
    去除缩放，令输入接受为 -1,1
    '''
    batch_size, channels, height, width = img.shape  # [NCHW]
    assert channels == 3
    x = img
    features = self.layers(x).reshape(-1, 2048)
    return features


InceptionV3.forward = patch_forward


class WrapNet(nn.Module):
    def __init__(self):
        super().__init__()
        net = InceptionV3()
        net.load_state_dict(torch.load('inceptionv3_weight.pt', 'cpu'))
        del net.output
        net.requires_grad_(False)
        net.eval()
        net.to('cpu')
        self.net = net

    def forward(self, x):
        y = self.net(x)
        return y
