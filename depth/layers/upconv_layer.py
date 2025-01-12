import torch
import torch.nn as nn
import torch.nn.functional as F
from wsconv_layer import conv_ws
from torch.jit import script

@script
def _mish_jit_fwd(x): return x.mul(torch.tanh(F.softplus(x)))

@script
def _mish_jit_bwd(x, grad_output):
    x_sigmoid = torch.sigmoid(x)
    x_tanh_sp = F.softplus(x).tanh()
    return grad_output.mul(x_tanh_sp + x * x_sigmoid * (1 - x_tanh_sp * x_tanh_sp))

class MishJitAutoFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return _mish_jit_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_variables[0]
        return _mish_jit_bwd(x, grad_output)

# Cell
def mish(x): return MishJitAutoFn.apply(x)

class Mish(nn.Module):
    def __init__(self, inplace: bool = False):
        super(Mish, self).__init__()

    def forward(self, x):
        return MishJitAutoFn.apply(x)
    
# pre-activation based upsampling conv block
class upConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor, norm, act, num_groups):
        super(upConvLayer, self).__init__()
        conv = conv_ws
        if act == 'ELU':
            act = nn.ELU()
        elif act == 'Mish':
            act = Mish()
        else:
            act = nn.ReLU(True)
        self.conv = conv(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        if norm == 'GN':
            self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels)
        else:
            self.norm = nn.BatchNorm2d(in_channels, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.act = act
        self.scale_factor = scale_factor
    def forward(self, x):
        x = self.norm(x)
        x = self.act(x)     #pre-activation
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        x = self.conv(x)
        return x

# pre-activation based conv block
class myConv(nn.Module):
    def __init__(self, in_ch, out_ch, kSize, stride=1, 
                    padding=0, dilation=1, bias=True, norm='GN', act='ELU', num_groups=32):
        super(myConv, self).__init__()
        conv = conv_ws
        if act == 'ELU':
            act = nn.ELU()
        elif act == 'Mish':
            act = Mish()
        else:
            act = nn.ReLU(True)
        module = []
        if norm == 'GN': 
            module.append(nn.GroupNorm(num_groups=num_groups, num_channels=in_ch))
        else:
            module.append(nn.BatchNorm2d(in_ch, eps=0.001, momentum=0.1, affine=True, track_running_stats=True))
        module.append(act)
        module.append(conv(in_ch, out_ch, kernel_size=kSize, stride=stride, 
                            padding=padding, dilation=dilation, groups=1, bias=bias))
        self.module = nn.Sequential(*module)
    def forward(self, x):
        out = self.module(x)
        return out
    

if __name__ == '__main__':
    ex = torch.rand(size=(2, 32, 32, 3)).cuda()
    layer = upConvLayer(in_channels=32, out_channels=3, scale_factor=1, norm='GN', act='ELU', num_groups=32).cuda()

    print(layer(ex).shape)