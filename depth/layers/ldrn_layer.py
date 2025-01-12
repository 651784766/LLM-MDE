from depth.layers.dpext_layer import *
from depth.layers.lapdecoder_layer import *

# Laplacian Depth Residual Network
class LDRN(nn.Module):
    def __init__(self, args):
        super(LDRN, self).__init__()
        lv6 = args.lv6
        encoder = args.encoder
        self.name = encoder
        if encoder == 'ResNext101':
            self.encoder = deepFeatureExtractor_ResNext101(args, lv6)
        elif encoder == 'VGG19':
            self.encoder = deepFeatureExtractor_VGG19(args, lv6)
        elif encoder == 'DenseNet161':
            self.encoder = deepFeatureExtractor_DenseNet161(args, lv6)
        elif encoder == 'InceptionV3':
            self.encoder = deepFeatureExtractor_InceptionV3(args, lv6)
        elif encoder == 'MobileNetV2':
            self.encoder = deepFeatureExtractor_MobileNetV2(args)
        elif encoder == 'ResNet101':
            self.encoder = deepFeatureExtractor_ResNet101(args, lv6)
        elif 'EfficientNet' in args.encoder:
            self.encoder = deepFeatureExtractor_EfficientNet(args, encoder, lv6)
        elif 'AST' in args.encoder:    
            self.encoder = deepFeatureExtractor_AST(args)

        if lv6 is True:
            self.decoder = Lap_decoder_lv6(args, self.encoder.dimList)
        else:
            self.decoder = Lap_decoder_lv5(args, self.encoder.dimList)
    def forward(self, x):
        #pdb.set_trace()
        if self.name == 'AST':
            out_featList, attention = self.encoder(x)
        else:
            out_featList = self.encoder(x)
            attention = None
        rgb_down2 = F.interpolate(x, scale_factor = 0.5, mode='bilinear', align_corners=False, recompute_scale_factor=True)
        rgb_down4 = F.interpolate(rgb_down2, scale_factor = 0.5, mode='bilinear', align_corners=False, recompute_scale_factor=True)
        rgb_down8 = F.interpolate(rgb_down4, scale_factor = 0.5, mode='bilinear', align_corners=False, recompute_scale_factor=True)
        rgb_down16 = F.interpolate(rgb_down8, scale_factor = 0.5, mode='bilinear', align_corners=False, recompute_scale_factor=True)
        rgb_down32 = F.interpolate(rgb_down16, scale_factor = 0.5, mode='bilinear', align_corners=False, recompute_scale_factor=True)
        rgb_up16 = F.interpolate(rgb_down32, rgb_down16.shape[2:], mode='bilinear', align_corners=False)
        rgb_up8 = F.interpolate(rgb_down16, rgb_down8.shape[2:], mode='bilinear', align_corners=False)
        rgb_up4 = F.interpolate(rgb_down8, rgb_down4.shape[2:], mode='bilinear', align_corners=False)
        rgb_up2 = F.interpolate(rgb_down4, rgb_down2.shape[2:], mode='bilinear', align_corners=False)
        rgb_up = F.interpolate(rgb_down2, x.shape[2:], mode='bilinear', align_corners=False)
        lap1 = x - rgb_up
        lap2 = rgb_down2 - rgb_up2
        lap3 = rgb_down4 - rgb_up4
        lap4 = rgb_down8 - rgb_up8
        lap5 = rgb_down16 - rgb_up16
        rgb_list = [rgb_down32, lap5, lap4, lap3, lap2, lap1]

        d_res_list, depth = self.decoder(out_featList, rgb_list)

        return depth   

    def train(self, mode=True):
        super().train(mode)
        self.encoder.freeze_bn()