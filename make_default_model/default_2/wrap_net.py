import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_fid.fid_score as fid_score
import pytorch_fid.inception


def patch_forward(self, inp):
    """Get Inception feature maps

    Parameters
    ----------
    inp : torch.autograd.Variable
        Input tensor of shape Bx3xHxW. Values are expected to be in
        range (0, 1)

    Returns
    -------
    List of torch.autograd.Variable, corresponding to the selected output
    block, sorted ascending by index
    """
    outp = []
    x = inp

    if self.resize_input:
        x = F.interpolate(x,
                          size=(299, 299),
                          mode='bilinear',
                          align_corners=False)

    if self.normalize_input:
        x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)

    is_complete = False
    for idx, block in enumerate(self.blocks):
        x = block(x)
        if idx in self.output_blocks and not is_complete:
            outp.append(x)

        if idx == self.last_needed_block:
            is_complete = True
            # break

    return outp


# make pytorch_fid fid_model can compile with jit.
pytorch_fid.inception.InceptionV3.forward = patch_forward


class WrapNet(nn.Module):
    def __init__(self):
        super().__init__()
        dims = 2048
        block_idx = fid_score.InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        net = fid_score.InceptionV3([block_idx], resize_input=False, normalize_input=False)

        net.requires_grad_(False)
        net.eval()
        net.to('cpu')
        self.net = net

    def forward(self, x):
        y = self.net(x)[0].squeeze(3).squeeze(2)
        return y
