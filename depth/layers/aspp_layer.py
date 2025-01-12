import torch
import torch.nn as nn
from depth.layers.wsconv_layer import conv_ws
from depth.layers.upconv_layer import myConv

# ASPP Module
class Dilated_bottleNeck(nn.Module):
    def __init__(self, norm, act, in_feat):
        super(Dilated_bottleNeck, self).__init__()
        conv = conv_ws
        # in feat = 1024 in ResNext101 and ResNet101
        self.reduction1 = conv(in_feat, in_feat//2, kernel_size=1, stride = 1, bias=False, padding=0)
        self.aspp_d3 = nn.Sequential(myConv(in_feat//2, in_feat//4, kSize=1, stride=1, padding=0, dilation=1,bias=False, norm=norm, act=act, num_groups=(in_feat//2)//16),
                                    myConv(in_feat//4, in_feat//4, kSize=3, stride=1, padding=3, dilation=3,bias=False, norm=norm, act=act, num_groups=(in_feat//4)//16))
        self.aspp_d6 = nn.Sequential(myConv(in_feat//2 + in_feat//4, in_feat//4, kSize=1, stride=1, padding=0, dilation=1,bias=False, norm=norm, act=act, num_groups=(in_feat//2 + in_feat//4)//16),
                                    myConv(in_feat//4, in_feat//4, kSize=3, stride=1, padding=6, dilation=6,bias=False, norm=norm, act=act, num_groups=(in_feat//4)//16))
        self.aspp_d12 = nn.Sequential(myConv(in_feat, in_feat//4, kSize=1, stride=1, padding=0, dilation=1,bias=False, norm=norm, act=act, num_groups=(in_feat)//16),
                                    myConv(in_feat//4, in_feat//4, kSize=3, stride=1, padding=12, dilation=12,bias=False, norm=norm, act=act, num_groups=(in_feat//4)//16))
        self.aspp_d18 = nn.Sequential(myConv(in_feat + in_feat//4, in_feat//4, kSize=1, stride=1, padding=0, dilation=1,bias=False, norm=norm, act=act, num_groups=(in_feat + in_feat//4)//16),
                                    myConv(in_feat//4, in_feat//4, kSize=3, stride=1, padding=18, dilation=18,bias=False, norm=norm, act=act, num_groups=(in_feat//4)//16))
        self.reduction2 = myConv(((in_feat//4)*4) + (in_feat//2), in_feat//2, kSize=3, stride=1, padding=1,bias=False, norm=norm, act=act, num_groups = ((in_feat//4)*4 + (in_feat//2))//16)
    
    def forward(self, x):
        x = self.reduction1(x)
        d3 = self.aspp_d3(x)
        cat1 = torch.cat([x, d3],dim=1)
        d6 = self.aspp_d6(cat1)
        cat2 = torch.cat([cat1, d6],dim=1)
        d12 = self.aspp_d12(cat2)
        cat3 = torch.cat([cat2, d12],dim=1)
        d18 = self.aspp_d18(cat3)
        out = self.reduction2(torch.cat([x,d3,d6,d12,d18], dim=1))
        return out      # 512 x H/16 x W/16

class Dilated_bottleNeck2(nn.Module):
    def __init__(self, norm, act, in_feat):
        super(Dilated_bottleNeck2, self).__init__()
        conv = conv_ws
        # in feat = 1024 in ResNext101 and ResNet101
        #self.reduction1 = conv(in_feat, in_feat//2, kernel_size=1, stride = 1, bias=False, padding=0)
        self.reduction1 = conv(in_feat, in_feat//2, kernel_size=3, stride = 1, padding=1, bias=False)
        self.aspp_d3 = nn.Sequential(myConv(in_feat//2, in_feat//4, kSize=1, stride=1, padding=0, dilation=1,bias=False, norm=norm, act=act, num_groups=(in_feat//2)//16),
                                    myConv(in_feat//4, in_feat//4, kSize=3, stride=1, padding=3, dilation=3,bias=False, norm=norm, act=act, num_groups=(in_feat//4)//16))
        self.aspp_d6 = nn.Sequential(myConv(in_feat//2 + in_feat//4, in_feat//4, kSize=1, stride=1, padding=0, dilation=1,bias=False, norm=norm, act=act, num_groups=(in_feat//2 + in_feat//4)//16),
                                    myConv(in_feat//4, in_feat//4, kSize=3, stride=1, padding=6, dilation=6,bias=False, norm=norm, act=act, num_groups=(in_feat//4)//16))
        self.aspp_d12 = nn.Sequential(myConv(in_feat, in_feat//4, kSize=1, stride=1, padding=0, dilation=1,bias=False, norm=norm, act=act, num_groups=(in_feat)//16),
                                    myConv(in_feat//4, in_feat//4, kSize=3, stride=1, padding=12, dilation=12,bias=False, norm=norm, act=act, num_groups=(in_feat//4)//16))
        self.aspp_d18 = nn.Sequential(myConv(in_feat + in_feat//4, in_feat//4, kSize=1, stride=1, padding=0, dilation=1,bias=False, norm=norm, act=act, num_groups=(in_feat + in_feat//4)//16),
                                    myConv(in_feat//4, in_feat//4, kSize=3, stride=1, padding=18, dilation=18,bias=False, norm=norm, act=act, num_groups=(in_feat//4)//16))
        self.aspp_d24 = nn.Sequential(myConv(in_feat + in_feat//2, in_feat//4, kSize=1, stride=1, padding=0, dilation=1,bias=False, norm=norm, act=act, num_groups=(in_feat + in_feat//2)//16),
                                    myConv(in_feat//4, in_feat//4, kSize=3, stride=1, padding=24, dilation=24,bias=False, norm=norm, act=act, num_groups=(in_feat//4)//16))
        self.reduction2 = myConv(((in_feat//4)*5) + (in_feat//2), in_feat//2, kSize=3, stride=1, padding=1,bias=False, norm=norm, act=act, num_groups = ((in_feat//4)*5 + (in_feat//2))//16)
    def forward(self, x):
        x = self.reduction1(x)
        d3 = self.aspp_d3(x)
        cat1 = torch.cat([x, d3],dim=1)
        d6 = self.aspp_d6(cat1)
        cat2 = torch.cat([cat1, d6],dim=1)
        d12 = self.aspp_d12(cat2)
        cat3 = torch.cat([cat2, d12],dim=1)
        d18 = self.aspp_d18(cat3)
        cat4 = torch.cat([cat3, d18],dim=1)
        d24 = self.aspp_d24(cat4)
        out = self.reduction2(torch.cat([x,d3,d6,d12,d18,d24], dim=1))
        return out      # 512 x H/16 x W/16

class Dilated_bottleNeck_lv6(nn.Module):
    def __init__(self, norm, act, in_feat):
        super(Dilated_bottleNeck_lv6, self).__init__()
        conv = conv_ws
        in_feat = in_feat//2
        self.reduction1 = myConv(in_feat*2, in_feat//2, kSize=3, stride=1, padding=1, bias=False, norm=norm, act=act, num_groups=(in_feat)//16)
        self.aspp_d3 = nn.Sequential(myConv(in_feat//2, in_feat//4, kSize=1, stride=1, padding=0, dilation=1,bias=False, norm=norm, act=act, num_groups=(in_feat//2)//16),
                                    myConv(in_feat//4, in_feat//4, kSize=3, stride=1, padding=3, dilation=3,bias=False, norm=norm, act=act, num_groups=(in_feat//4)//16))
        self.aspp_d6 = nn.Sequential(myConv(in_feat//2 + in_feat//4, in_feat//4, kSize=1, stride=1, padding=0, dilation=1,bias=False, norm=norm, act=act, num_groups=(in_feat//2 + in_feat//4)//16),
                                    myConv(in_feat//4, in_feat//4, kSize=3, stride=1, padding=6, dilation=6,bias=False, norm=norm, act=act, num_groups=(in_feat//4)//16))
        self.aspp_d12 = nn.Sequential(myConv(in_feat, in_feat//4, kSize=1, stride=1, padding=0, dilation=1,bias=False, norm=norm, act=act, num_groups=(in_feat)//16),
                                    myConv(in_feat//4, in_feat//4, kSize=3, stride=1, padding=12, dilation=12,bias=False, norm=norm, act=act, num_groups=(in_feat//4)//16))
        self.aspp_d18 = nn.Sequential(myConv(in_feat + in_feat//4, in_feat//4, kSize=1, stride=1, padding=0, dilation=1,bias=False, norm=norm, act=act, num_groups=(in_feat + in_feat//4)//16),
                                    myConv(in_feat//4, in_feat//4, kSize=3, stride=1, padding=18, dilation=18,bias=False, norm=norm, act=act, num_groups=(in_feat//4)//16))
        self.reduction2 = myConv(((in_feat//4)*4) + (in_feat//2), in_feat, kSize=3, stride=1, padding=1,bias=False, norm=norm, act=act, num_groups = ((in_feat//4)*4 + (in_feat//2))//16)
    def forward(self, x):
        x = self.reduction1(x)
        d3 = self.aspp_d3(x)
        cat1 = torch.cat([x, d3],dim=1)
        d6 = self.aspp_d6(cat1)
        cat2 = torch.cat([cat1, d6],dim=1)
        d12 = self.aspp_d12(cat2)
        cat3 = torch.cat([cat2, d12],dim=1)
        d18 = self.aspp_d18(cat3)
        out = self.reduction2(torch.cat([x,d3,d6,d12,d18], dim=1))
        return out      # 512 x H/16 x W/16