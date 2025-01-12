import torch
import torch.nn as nn
import torch.nn.functional as F
from depth.layers.aspp_layer import Dilated_bottleNeck, Dilated_bottleNeck_lv6
from depth.layers.upconv_layer import myConv, upConvLayer
from depth.layers.wsconv_layer import conv_ws

# Laplacian Decoder Network
class Lap_decoder_lv5(nn.Module):
    def __init__(self, args, dimList):
        super(Lap_decoder_lv5, self).__init__()
        norm = args.norm
        conv = conv_ws
        # if norm == 'GN':
        #     if args.local_rank == 0:
        #         print("==> Norm: GN")
        # else:
        #     if args.local_rank == 0:
        #         print("==> Norm: BN")
            
        if args.act == 'ELU':
            act = 'ELU'
        elif args.act == 'Mish':
            act = 'Mish'
        else:
            act = 'ReLU'
        kSize = 3
        self.max_depth = args.max_depth
        self.ASPP = Dilated_bottleNeck(norm, act, dimList[3])
        self.dimList = dimList
        ############################################     Pyramid Level 5     ###################################################
        # decoder1 out : 1 x H/16 x W/16 (Level 5)
        self.decoder1 = nn.Sequential(myConv(dimList[3]//2, dimList[3]//4, kSize, stride=1, padding=kSize//2, bias=False, 
                                            norm=norm, act=act, num_groups=(dimList[3]//2)//16),      
                                        myConv(dimList[3]//4, dimList[3]//8, kSize, stride=1, padding=kSize//2, bias=False, 
                                            norm=norm, act=act, num_groups=(dimList[3]//4)//16),    
                                        myConv(dimList[3]//8, dimList[3]//16, kSize, stride=1, padding=kSize//2, bias=False, 
                                            norm=norm, act=act, num_groups=(dimList[3]//8)//16),  
                                        myConv(dimList[3]//16, dimList[3]//32, kSize, stride=1, padding=kSize//2, bias=False, 
                                            norm=norm, act=act, num_groups=(dimList[3]//16)//16),
                                        myConv(dimList[3]//32, 1, kSize, stride=1, padding=kSize//2, bias=False, 
                                            norm=norm, act=act, num_groups=(dimList[3]//32)//16)
                                     )
        ########################################################################################################################

        ############################################     Pyramid Level 4     ###################################################
        # decoder2 out : 1 x H/8 x W/8 (Level 4)
        # decoder2_up : (H/16,W/16)->(H/8,W/8)
        self.decoder2_up1 = upConvLayer(dimList[3]//2, dimList[3]//4, 2, norm, act, (dimList[3]//2)//16)
        self.decoder2_reduc1 = myConv(dimList[3]//4 + dimList[2], dimList[3]//4 - 4, kSize=1, stride=1, padding=0,bias=False, 
                                        norm=norm, act=act, num_groups = (dimList[3]//4 + dimList[2])//16)
        self.decoder2_1 = myConv(dimList[3]//4, dimList[3]//4, kSize, stride=1, padding=kSize//2, bias=False, 
                                        norm=norm, act=act, num_groups=(dimList[3]//4)//16)
        
        self.decoder2_2 = myConv(dimList[3]//4, dimList[3]//8, kSize, stride=1, padding=kSize//2, bias=False, 
                                        norm=norm, act=act, num_groups=(dimList[3]//4)//16)
        self.decoder2_3 = myConv(dimList[3]//8, dimList[3]//16, kSize, stride=1, padding=kSize//2, bias=False, 
                                        norm=norm, act=act, num_groups=(dimList[3]//8)//16)
        
        self.decoder2_4 = myConv(dimList[3]//16, 1, kSize, stride=1, padding=kSize//2, bias=False, 
                                        norm=norm, act=act, num_groups=(dimList[3]//16)//16)
        ########################################################################################################################

        ############################################     Pyramid Level 3     ###################################################
        # decoder2 out2 : 1 x H/4 x W/4 (Level 3)
        # decoder2_1_up2 : (H/8,W/8)->(H/4,W/4)
        self.decoder2_1_up2 = upConvLayer(dimList[3]//4, dimList[3]//8, 2, norm, act, (dimList[3]//4)//16)
        self.decoder2_1_reduc2 = myConv(dimList[3]//8 + dimList[1], dimList[3]//8 - 4, kSize=1, stride=1, padding=0,bias=False, 
                                        norm=norm, act=act, num_groups = (dimList[3]//8 + dimList[1])//16)
        self.decoder2_1_1 = myConv(dimList[3]//8, dimList[3]//8, kSize, stride=1, padding=kSize//2, bias=False, 
                                        norm=norm, act=act, num_groups=(dimList[3]//8)//16)
        
        self.decoder2_1_2 = myConv(dimList[3]//8, dimList[3]//16, kSize, stride=1, padding=kSize//2, bias=False, 
                                        norm=norm, act=act, num_groups=(dimList[3]//8)//16)
        
        self.decoder2_1_3 = myConv(dimList[3]//16, 1, kSize, stride=1, padding=kSize//2, bias=False, 
                                        norm=norm, act=act, num_groups=(dimList[3]//16)//16)
        ########################################################################################################################

        ############################################     Pyramid Level 2     ###################################################
        # decoder2 out3 : 1 x H/2 x W/2 (Level 2)
        # decoder2_1_1_up3 : (H/4,W/4)->(H/2,W/2)
        self.decoder2_1_1_up3 = upConvLayer(dimList[3]//8, dimList[3]//16, 2, norm, act, (dimList[3]//8)//16)
        self.decoder2_1_1_reduc3 = myConv(dimList[3]//16 + dimList[0], dimList[3]//16 - 4, kSize=1, stride=1, padding=0,bias=False, 
                                        norm=norm, act=act, num_groups = (dimList[3]//16 + dimList[0])//16)
        self.decoder2_1_1_1 = myConv(dimList[3]//16, dimList[3]//16, kSize, stride=1, padding=kSize//2, bias=False, 
                                        norm=norm, act=act, num_groups=(dimList[3]//16)//16)
        
        self.decoder2_1_1_2 = myConv(dimList[3]//16, 1, kSize, stride=1, padding=kSize//2, bias=False, 
                                        norm=norm, act=act, num_groups=(dimList[3]//16)//16)
        ########################################################################################################################
        
        ############################################     Pyramid Level 1     ###################################################
        # decoder5 out : 1 x H x W (Level 1)
        # decoder2_1_1_1_up4 : (H/2,W/2)->(H,W)
        self.decoder2_1_1_1_up4 = upConvLayer(dimList[3]//16, dimList[3]//16 - 4, 2, norm, act, (dimList[3]//16)//16)
        self.decoder2_1_1_1_1 = myConv(dimList[3]//16, dimList[3]//16, kSize, stride=1, padding=kSize//2, bias=False, 
                                        norm=norm, act=act, num_groups=(dimList[3]//16)//16)
        
        self.decoder2_1_1_1_2 = myConv(dimList[3]//16, dimList[3]//32, kSize, stride=1, padding=kSize//2, bias=False, 
                                        norm=norm, act=act, num_groups=(dimList[3]//16)//16)
        self.decoder2_1_1_1_3 = myConv(dimList[3]//32, 1, kSize, stride=1, padding=kSize//2, bias=False, 
                                        norm=norm, act=act, num_groups=(dimList[3]//32)//16)
        ########################################################################################################################
        self.upscale = F.interpolate

    def forward(self, x, rgb):
        cat1, cat2, cat3, dense_feat = x[0], x[1], x[2], x[3]
        rgb_lv6, rgb_lv5, rgb_lv4, rgb_lv3, rgb_lv2, rgb_lv1 = rgb[0], rgb[1], rgb[2], rgb[3], rgb[4], rgb[5]
        dense_feat = self.ASPP(dense_feat)                        # Dense feature for lev 5
        # decoder 1 - Pyramid level 5
        lap_lv5 = torch.sigmoid(self.decoder1(dense_feat))
        lap_lv5_up = self.upscale(lap_lv5, scale_factor = 2, mode='bilinear', align_corners=False)

        # decoder 2 - Pyramid level 4
        dec2 = self.decoder2_up1(dense_feat)
        dec2 = self.decoder2_reduc1(torch.cat([dec2,cat3],dim=1))
        dec2_up = self.decoder2_1(torch.cat([dec2,lap_lv5_up,rgb_lv4],dim=1))
        dec2 = self.decoder2_2(dec2_up)
        dec2 = self.decoder2_3(dec2)
        lap_lv4 = torch.tanh(self.decoder2_4(dec2) + (0.1*rgb_lv4.mean(dim=1,keepdim=True)))                 
        # if depth range is (0,1), laplacian of image range is (-1,1)
        lap_lv4_up = self.upscale(lap_lv4, scale_factor = 2, mode='bilinear', align_corners=False)
        # decoder 2 - Pyramid level 3
        dec3 = self.decoder2_1_up2(dec2_up)
        dec3 = self.decoder2_1_reduc2(torch.cat([dec3,cat2],dim=1))
        dec3_up = self.decoder2_1_1(torch.cat([dec3,lap_lv4_up,rgb_lv3],dim=1))
        dec3 = self.decoder2_1_2(dec3_up)
        lap_lv3 = torch.tanh(self.decoder2_1_3(dec3) + (0.1*rgb_lv3.mean(dim=1,keepdim=True)))                 
        # if depth range is (0,1), laplacian of image range is (-1,1)
        lap_lv3_up = self.upscale(lap_lv3, scale_factor = 2, mode='bilinear', align_corners=False)
        # decoder 2 - Pyramid level 2
        dec4 = self.decoder2_1_1_up3(dec3_up)
        dec4 = self.decoder2_1_1_reduc3(torch.cat([dec4,cat1],dim=1))
        dec4_up = self.decoder2_1_1_1(torch.cat([dec4,lap_lv3_up,rgb_lv2],dim=1))

        lap_lv2 = torch.tanh(self.decoder2_1_1_2(dec4_up) + (0.1*rgb_lv2.mean(dim=1,keepdim=True)))                  
        # if depth range is (0,1), laplacian of image range is (-1,1)
        lap_lv2_up = self.upscale(lap_lv2, scale_factor = 2, mode='bilinear', align_corners=False)
        # decoder 2 - Pyramid level 1
        dec5 = self.decoder2_1_1_1_up4(dec4_up)
        dec5 = self.decoder2_1_1_1_1(torch.cat([dec5,lap_lv2_up,rgb_lv1],dim=1))
        dec5 = self.decoder2_1_1_1_2(dec5)
        lap_lv1 = torch.tanh(self.decoder2_1_1_1_3(dec5) + (0.1*rgb_lv1.mean(dim=1,keepdim=True)))
        # if depth range is (0,1), laplacian of image range is (-1,1)
        
        # Laplacian restoration
        lap_lv4_img = lap_lv4 + lap_lv5_up
        lap_lv3_img = lap_lv3 + self.upscale(lap_lv4_img, scale_factor = 2,mode='bilinear', align_corners=False)
        lap_lv2_img = lap_lv2 + self.upscale(lap_lv3_img, scale_factor = 2,mode='bilinear', align_corners=False)
        final_depth = lap_lv1 + self.upscale(lap_lv2_img, scale_factor = 2,mode='bilinear', align_corners=False)
        final_depth = torch.sigmoid(final_depth)
        return [(lap_lv5)*self.max_depth, (lap_lv4)*self.max_depth, (lap_lv3)*self.max_depth, (lap_lv2)*self.max_depth, (lap_lv1)*self.max_depth], final_depth*self.max_depth
        # fit laplacian image range (-80,80), depth image range(0,80)

class Lap_decoder_lv6(nn.Module):
    def __init__(self, args, dimList):
        super(Lap_decoder_lv6, self).__init__()
        norm = args.norm
        conv = conv_ws
        if norm == 'GN':
            if args.local_rank == 0:
                print("==> Norm: GN")
        else:
            if args.local_rank == 0:
                print("==> Norm: BN")

        if args.act == 'ELU':
            act = 'ELU'
        elif args.act == 'Mish':
            act = 'Mish'
        else:
            act = 'ReLU'
        kSize = 3
        self.max_depth = args.max_depth
        self.ASPP = Dilated_bottleNeck_lv6(norm, act, dimList[4])
        dimList[4] = dimList[4]//2  
        self.dimList = dimList
        ############################################     Pyramid Level 6     ###################################################
        # decoder1 out : 1 x H/32 x W/32 (Level 6)
        self.decoder1 = nn.Sequential(myConv(dimList[4]//2, dimList[4]//4, kSize, stride=1, padding=kSize//2, bias=False, 
                                                norm=norm, act=act, num_groups=(dimList[4]//2)//16),
                                        myConv(dimList[4]//4, dimList[4]//8, kSize, stride=1, padding=kSize//2, bias=False, 
                                                norm=norm, act=act, num_groups=(dimList[4]//4)//16),
                                        myConv(dimList[4]//8, dimList[4]//16, kSize, stride=1, padding=kSize//2, bias=False, 
                                                norm=norm, act=act, num_groups=(dimList[4]//8)//16),
                                        myConv(dimList[4]//16, dimList[4]//32, kSize, stride=1, padding=kSize//2, bias=False, 
                                                norm=norm, act=act, num_groups=(dimList[4]//16)//16),
                                        myConv(dimList[4]//32, 1, kSize, stride=1, padding=kSize//2, bias=False, 
                                                norm=norm, act=act, num_groups=(dimList[4]//32)//8)
                                     )
        ########################################################################################################################

        ############################################     Pyramid Level 5     ###################################################
        # decoder2 out : 1 x H/16 x W/16 (Level 5)
        # decoder2_up : (H/32,W/32)->(H/16,W/16)
        self.decoder2_up1 = upConvLayer(dimList[4]//2, dimList[4]//4, 2, norm, act, (dimList[4]//2)//16)
        self.decoder2_reduc1 = myConv(dimList[4]//4 + dimList[3], dimList[4]//4 - 4, kSize=1, stride=1, padding=0, bias=False, 
                                            norm=norm, act=act, num_groups = (dimList[4]//4 + dimList[3])//16)
        self.decoder2_1 = myConv(dimList[4]//4, dimList[4]//4, kSize, stride=1, padding=kSize//2, bias=False, 
                                            norm=norm, act=act, num_groups=(dimList[4]//4)//16)
        
        self.decoder2_2 = myConv(dimList[4]//4, dimList[4]//8, kSize, stride=1, padding=kSize//2, bias=False, 
                                            norm=norm, act=act, num_groups=(dimList[4]//4)//16)
        self.decoder2_3 = myConv(dimList[4]//8, dimList[4]//16, kSize, stride=1, padding=kSize//2, bias=False, 
                                            norm=norm, act=act, num_groups=(dimList[4]//8)//16)
        
        self.decoder2_4 = myConv(dimList[4]//16, 1, kSize, stride=1, padding=kSize//2, bias=False, 
                                            norm=norm, act=act, num_groups=(dimList[4]//16)//16)
        ########################################################################################################################
        
        ############################################     Pyramid Level 4     ###################################################
        # decoder2 out2 : 1 x H/8 x W/8 (Level 4)
        # decoder2_1_up2 : (H/16,W/16)->(H/8,W/8)
        self.decoder2_1_up2 = upConvLayer(dimList[4]//4, dimList[4]//8, 2, norm, act, (dimList[4]//4)//16)
        self.decoder2_1_reduc2 = myConv(dimList[4]//8 + dimList[2], dimList[4]//8 - 4, kSize=1, stride=1, padding=0, bias=False, 
                                            norm=norm, act=act, num_groups = (dimList[4]//8 + dimList[2])//16)
        self.decoder2_1_1 = myConv(dimList[4]//8, dimList[4]//8, kSize, stride=1, padding=kSize//2, bias=False, 
                                            norm=norm, act=act, num_groups=(dimList[4]//8)//16)
        
        self.decoder2_1_2 = myConv(dimList[4]//8, dimList[4]//16, kSize, stride=1, padding=kSize//2, bias=False, 
                                            norm=norm, act=act, num_groups=(dimList[4]//8)//16)
        
        self.decoder2_1_3 = myConv(dimList[4]//16, 1, kSize, stride=1, padding=kSize//2, bias=False, 
                                            norm=norm, act=act, num_groups=(dimList[4]//16)//16)
        ########################################################################################################################

        ############################################     Pyramid Level 3     ###################################################
        # decoder2 out3 : 1 x H/4 x W/4 (Level 3)
        # decoder2_1_1_up3 : (H/8,W/8)->(H/4,W/4)
        self.decoder2_1_1_up3 = upConvLayer(dimList[4]//8, dimList[4]//16, 2, norm, act, (dimList[4]//8)//16)
        self.decoder2_1_1_reduc3 = myConv(dimList[4]//16 + dimList[1], dimList[4]//16 - 4, kSize=1, stride=1, padding=0,bias=False, 
                                             norm=norm, act=act, num_groups = (dimList[4]//16 + dimList[1])//8)
        self.decoder2_1_1_1 = myConv(dimList[4]//16, dimList[4]//16, kSize, stride=1, padding=kSize//2, bias=False, 
                                             norm=norm, act=act, num_groups=(dimList[4]//16)//16)
        
        self.decoder2_1_1_2 = myConv(dimList[4]//16, 1, kSize, stride=1, padding=kSize//2, bias=False, 
                                             norm=norm, act=act, num_groups=(dimList[4]//16)//16)
        ########################################################################################################################

        ############################################     Pyramid Level 2     ###################################################
        # decoder2 out4 : 1 x H/2 x W/2 (Level 2)
        # decoder2_1_1_1_up4 : (H/4,W/4)->(H/2,W/2)
        self.decoder2_1_1_1_up4 = upConvLayer(dimList[4]//16, dimList[4]//32, 2, norm, act, (dimList[4]//16)//16)
        self.decoder2_1_1_1_reduc4 = myConv(dimList[4]//32 + dimList[0], dimList[4]//32 - 4, kSize=1, stride=1, padding=0, bias=False, 
                                              norm=norm, act=act, num_groups=(dimList[4]//32 + dimList[0])//8)
        self.decoder2_1_1_1_1 = myConv(dimList[4]//32, dimList[4]//32, kSize, stride=1, padding=kSize//2, bias=False, 
                                              norm=norm, act=act, num_groups=(dimList[4]//32)//8)

        self.decoder2_1_1_1_2 = myConv(dimList[4]//32, 1, kSize, stride=1, padding=kSize//2, bias=False, 
                                              norm=norm, act=act, num_groups=(dimList[4]//32)//8)
        ########################################################################################################################

        ############################################     Pyramid Level 1     ###################################################
        # decoder5 out : 1 x H x W (Level 1)
        # decoder2_1_1_1_1_up5 : (H/2,W/2)->(H,W)
        self.decoder2_1_1_1_1_up5 = upConvLayer(dimList[4]//32, dimList[4]//32 - 4, 2, norm, act, (dimList[4]//32)//8) # H x W (64 -> 60)
        self.decoder2_1_1_1_1_1 = myConv(dimList[4]//32, dimList[4]//32, kSize, stride=1, padding=kSize//2, bias=False, 
                                              norm=norm, act=act, num_groups=(dimList[4]//32)//8)
        
        self.decoder2_1_1_1_1_2 = myConv(dimList[4]//32, dimList[4]//64, kSize, stride=1, padding=kSize//2, bias=False, 
                                              norm=norm, act=act, num_groups=(dimList[4]//32)//8)
        self.decoder2_1_1_1_1_3 = myConv(dimList[4]//64, 1, kSize, stride=1, padding=kSize//2, bias=False, 
                                              norm=norm, act=act, num_groups=(dimList[4]//64)//4)
        ########################################################################################################################
        self.upscale = F.interpolate

    def forward(self, x, rgb):
        cat1, cat2, cat3, cat4, dense_feat = x[0], x[1], x[2], x[3], x[4]
        rgb_lv6, rgb_lv5, rgb_lv4, rgb_lv3, rgb_lv2, rgb_lv1 = rgb[0], rgb[1], rgb[2], rgb[3], rgb[4], rgb[5]
        dense_feat = self.ASPP(dense_feat)                        # Dense feature for lev 6
        # decoder 1 - Pyramid level 6
        lap_lv6 = torch.sigmoid(self.decoder1(dense_feat))
        lap_lv6_up = self.upscale(lap_lv6, scale_factor = 2, mode='bilinear', align_corners=False)

        # decoder 2 - Pyramid level 5
        dec2 = self.decoder2_up1(dense_feat)
        dec2 = self.decoder2_reduc1(torch.cat([dec2,cat4],dim=1))
        dec2_up = self.decoder2_1(torch.cat([dec2,lap_lv6_up,rgb_lv5],dim=1))
        dec2 = self.decoder2_2(dec2_up)
        dec2 = self.decoder2_3(dec2)
        lap_lv5 = torch.tanh(self.decoder2_4(dec2) + (0.1*rgb_lv5.mean(dim=1,keepdim=True)))              
        # if depth range is (0,1), laplacian image range is (-1,1)
        lap_lv5_up = self.upscale(lap_lv5, scale_factor = 2, mode='bilinear', align_corners=False)
        # decoder 2 - Pyramid level 4
        dec3 = self.decoder2_1_up2(dec2_up)
        dec3 = self.decoder2_1_reduc2(torch.cat([dec3,cat3],dim=1))
        dec3_up = self.decoder2_1_1(torch.cat([dec3,lap_lv5_up,rgb_lv4],dim=1))
        dec3 = self.decoder2_1_2(dec3_up)
        lap_lv4 = torch.tanh(self.decoder2_1_3(dec3) + (0.1*rgb_lv4.mean(dim=1,keepdim=True)))                 
        # if depth range is (0,1), laplacian image range is (-1,1)
        lap_lv4_up = self.upscale(lap_lv4, scale_factor = 2, mode='bilinear', align_corners=False)
        # decoder 2 - Pyramid level 3
        dec4 = self.decoder2_1_1_up3(dec3_up)
        dec4 = self.decoder2_1_1_reduc3(torch.cat([dec4,cat2],dim=1))
        dec4_up = self.decoder2_1_1_1(torch.cat([dec4,lap_lv4_up,rgb_lv3],dim=1))

        lap_lv3 = torch.tanh(self.decoder2_1_1_2(dec4_up) + (0.1*rgb_lv3.mean(dim=1,keepdim=True)))                  
        # if depth range is (0,1), laplacian image range is (-1,1)
        lap_lv3_up = self.upscale(lap_lv3, scale_factor = 2, mode='bilinear', align_corners=False)
        # decoder 2 - Pyramid level 2
        dec5 = self.decoder2_1_1_1_up4(dec4_up)
        dec5 = self.decoder2_1_1_1_reduc4(torch.cat([dec5, cat1],dim=1))
        dec5_up = self.decoder2_1_1_1_1(torch.cat([dec5,lap_lv3_up,rgb_lv2],dim=1))

        lap_lv2 = torch.tanh(self.decoder2_1_1_1_2(dec5_up) + (0.1*rgb_lv2.mean(dim=1,keepdim=True)))
        # if depth range is (0,1), laplacian image range is (-1,1)
        lap_lv2_up = self.upscale(lap_lv2, scale_factor =2, mode='bilinear', align_corners=False)
        # decoder 2 - Pyramid level 1
        dec6 = self.decoder2_1_1_1_1_up5(dec5_up)
        dec6 = self.decoder2_1_1_1_1_1(torch.cat([dec6,lap_lv2_up,rgb_lv1],dim=1))
        dec6 = self.decoder2_1_1_1_1_2(dec6)
        lap_lv1 = torch.tanh(self.decoder2_1_1_1_1_3(dec6) + (0.1*rgb_lv1.mean(dim=1,keepdim=True)))                 
        # if depth range is (0,1), laplacian image range is (-1,1)

        # Laplacian restoration
        lap_lv5_img = lap_lv5 + lap_lv6_up
        lap_lv4_img = lap_lv4 + self.upscale(lap_lv5_img, scale_factor = 2,mode='bilinear', align_corners=False)
        lap_lv3_img = lap_lv3 + self.upscale(lap_lv4_img, scale_factor = 2,mode='bilinear', align_corners=False)
        lap_lv2_img = lap_lv2 + self.upscale(lap_lv3_img, scale_factor = 2,mode='bilinear', align_corners=False)
        final_depth = lap_lv1 + self.upscale(lap_lv2_img, scale_factor = 2,mode='bilinear', align_corners=False)
        final_depth = torch.sigmoid(final_depth)
        return [(lap_lv6)*self.max_depth, (lap_lv5)*self.max_depth, (lap_lv4)*self.max_depth, (lap_lv3)*self.max_depth, (lap_lv2)*self.max_depth, (lap_lv1)*self.max_depth], final_depth*self.max_depth
        # fit laplacian image range (-80,80), depth image range(0,80)