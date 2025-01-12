import torch
import torch.nn as nn
import torch.nn.functional as F
import geffnet
import torchvision.models as models
from depth.models import ast_transformer as model

class deepFeatureExtractor_AST(nn.Module):
    def __init__(self,args, lv6 = False):
        super(deepFeatureExtractor_AST, self).__init__()
        self.args = args
        # after passing ReLU   : H/2  x W/2
        # after passing Layer1 : H/4  x W/4
        # after passing Layer2 : H/8  x W/8
        # after passing Layer3 : H/16 x W/16
        #pdb.set_trace()
        if args.dataset == 'NYU':
            self.encoder = model.ASTransformer(args,size = (416,544),  patch_embed='conv')
        elif args.dataset == 'KITTI':
            self.encoder = model.ASTransformer(args,size = (352,704),  patch_embed='conv')
        freeze = True
        if freeze:
            for name, param in self.encoder.model.named_parameters():
                # param.requires_grad = False
                # if args.local_rank==0:
                #     print('Freeze '+name)
                if 'patch_embed' in name:
                    param.requires_grad = False
                    print('Freeze '+name)

        
        self.dimList = [96 + 12, 192 + 48, 384 + 192, 768]
        self.split = nn.Unfold(kernel_size=(2, 2), stride=(2, 2))
        self.conv_out1 = nn.Conv2d(12,12,kernel_size=(1, 1), stride=(1, 1))
        self.conv_out2 = nn.Conv2d(48,48,kernel_size=(1, 1), stride=(1, 1))
        self.conv_out3 = nn.Conv2d(192,192,kernel_size=(1, 1), stride=(1, 1))
        #self.dimList = [96, 192, 384, 768]
        self.conv1 =  nn.Conv2d(self.dimList[0],self.dimList[0],kernel_size=3,stride=1,padding=1,bias=False)
        self.conv2 =  nn.Conv2d(self.dimList[1],self.dimList[1],kernel_size=3,stride=1,padding=1,bias=False)
        self.conv3 =  nn.Conv2d(self.dimList[2],self.dimList[2],kernel_size=3,stride=1,padding=1,bias=False)
        self.conv4 =  nn.Conv2d(self.dimList[3],self.dimList[3],kernel_size=3,stride=1,padding=1,bias=False)


    def forward(self, x):
        B,C,H,W = x.shape
        img1 = self.split(x).reshape(B,-1,H//2,W//2)
        img2 = self.split(img1).reshape(B,-1,H//4,W//4)
        img3 = self.split(img2).reshape(B,-1,H//8,W//8)
        out_featList, attention, AUB_feature = self.encoder(x,[img1,img2,img3])
        feature1 = self.conv_out1(AUB_feature[0])
        feature2 = self.conv_out2(AUB_feature[1])
        feature3 = self.conv_out3(AUB_feature[2])
        out_featList[0] = self.conv1(torch.cat([out_featList[0],feature1],1))
        out_featList[1] = self.conv2(torch.cat([out_featList[1],feature2],1))
        out_featList[2] = self.conv3(torch.cat([out_featList[2],feature3],1))
        out_featList[3] = self.conv4(out_featList[3])
        #pdb.set_trace()
        return out_featList, attention

    def freeze_bn(self, enable=False):
        """ Adapted from https://discuss.pytorch.org/t/how-to-train-with-frozen-batchnorm/12106/8 """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train() if enable else module.eval()

                module.weight.requires_grad = enable
                module.bias.requires_grad = enable

class deepFeatureExtractor_ResNext101(nn.Module):
    def __init__(self,args, lv6 = False):
        super(deepFeatureExtractor_ResNext101, self).__init__()
        self.args = args
        # after passing ReLU   : H/2  x W/2
        # after passing Layer1 : H/4  x W/4
        # after passing Layer2 : H/8  x W/8
        # after passing Layer3 : H/16 x W/16
        self.encoder = models.resnext101_32x8d(pretrained=True)
        self.fixList = ['layer1.0','layer1.1','.bn']
        self.lv6 = lv6

        if lv6 is True:
            self.layerList = ['relu','layer1','layer2','layer3', 'layer4']
            self.dimList = [64, 256, 512, 1024,2048]
        else:
            del self.encoder.layer4
            del self.encoder.fc
            self.layerList = ['relu','layer1','layer2','layer3']
            self.dimList = [64, 256, 512, 1024]

        for name, parameters in self.encoder.named_parameters():
            if name == 'conv1.weight':
                parameters.requires_grad = False
            if any(x in name for x in self.fixList):
                parameters.requires_grad = False
        
    def forward(self, x):
        out_featList = []
        feature = x
        for k, v in self.encoder._modules.items():
            if k == 'avgpool':
                break
            feature = v(feature)
            #feature = v(features[-1])
            #features.append(feature)
            if any(x in k for x in self.layerList):
                out_featList.append(feature)
        return out_featList
    def freeze_bn(self, enable=False):
        """ Adapted from https://discuss.pytorch.org/t/how-to-train-with-frozen-batchnorm/12106/8 """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train() if enable else module.eval()

                module.weight.requires_grad = enable
                module.bias.requires_grad = enable

class deepFeatureExtractor_VGG19(nn.Module):
    def __init__(self,args):
        super(deepFeatureExtractor_VGG19, self).__init__()
        self.args = args
        # after passing 6th layer   : H/2  x W/2
        # after passing 13th layer   : H/4  x W/4
        # after passing 26th layer   : H/8  x W/8
        # after passing 39th layer   : H/16 x W/16
        # after passing 52th layer   : H/32 x W/32
        self.encoder = models.vgg19_bn(pretrained=True)
        del self.encoder.avgpool
        del self.encoder.classifier
        if lv6 is True:  # type: ignore
            self.dimList = [64, 128, 256, 512, 512]
            self.layerList = [6, 13, 26, 39, 52]
        else:
            self.dimList = [64, 128, 256, 512]
            self.layerList = [6, 13, 26, 39]
            for i in range(13):
                del self.encoder.features[-1]
        '''
        self.fixList = ['.bn']
        for name, parameters in self.encoder.named_parameters():
            if any(x in name for x in self.fixList):
                parameters.requires_grad = False
        '''
    def forward(self, x):
        out_featList = []
        feature = x
        for i in range(len(self.encoder.features)):
            feature = self.encoder.features[i](feature)
            if i in self.layerList:
                out_featList.append(feature)
        return out_featList
    def freeze_bn(self, enable=False):
        """ Adapted from https://discuss.pytorch.org/t/how-to-train-with-frozen-batchnorm/12106/8 """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train() if enable else module.eval()

                module.weight.requires_grad = enable
                module.bias.requires_grad = enable

class deepFeatureExtractor_DenseNet161(nn.Module):
    def __init__(self,args, lv6 = False):
        super(deepFeatureExtractor_DenseNet161, self).__init__()
        self.args = args
        self.encoder = models.densenet161(pretrained=True)
        self.lv6 = lv6
        del self.encoder.classifier
        del self.encoder.features.norm5
        if lv6 is True:
            self.dimList = [96, 192, 384, 1056, 2208]
        else:
            self.dimList = [96, 192, 384, 1056]
            del self.encoder.features.denseblock4

    def forward(self, x):
        out_featList = []
        feature = x
        for k, v in self.encoder.features._modules.items():
            if ('transition' in k):
                feature = v.norm(feature)
                feature = v.relu(feature)
                feature = v.conv(feature)
                out_featList.append(feature)
                feature = v.pool(feature)
            elif k == 'conv0':
                feature = v(feature)
                out_featList.append(feature)
            elif k == 'denseblock4' and (self.lv6 is True):
                feature = v(feature)
                out_featList.append(feature)
            else:
                feature = v(feature)
        return out_featList

    def freeze_bn(self, enable=False):
        """ Adapted from https://discuss.pytorch.org/t/how-to-train-with-frozen-batchnorm/12106/8 """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train() if enable else module.eval()

                module.weight.requires_grad = enable
                module.bias.requires_grad = enable

class deepFeatureExtractor_InceptionV3(nn.Module):
    def __init__(self,args, lv6 = False):
        super(deepFeatureExtractor_InceptionV3, self).__init__()
        self.args = args
        self.encoder = models.inception_v3(pretrained=True)
        self.encoder.aux_logits = False
        self.lv6 = lv6
        del self.encoder.AuxLogits
        del self.encoder.fc
        if lv6 is True:
            self.layerList = ['Conv2d_2b_3x3','Conv2d_4a_3x3','Mixed_5d','Mixed_6e','Mixed_7c']
            self.dimList = [64, 192, 288, 768, 2048]
        else:
            self.layerList = ['Conv2d_2b_3x3','Conv2d_4a_3x3','Mixed_5d','Mixed_6e']
            self.dimList = [64, 192, 288, 768]
            del self.encoder.Mixed_7a
            del self.encoder.Mixed_7b
            del self.encoder.Mixed_7c

    def forward(self, x):
        out_featList = []
        feature = x
        for k, v in self.encoder._modules.items():
            feature = v(feature)
            if k in ['Conv2d_2b_3x3', 'Conv2d_ta_3x3']:
                feature = F.max_pool2d(feature, kernel_size=3, stride=2)
            if any(x in k for x in self.layerList):
                out_featList.append(feature)
        return out_featList

    def freeze_bn(self, enable=False):
        """ Adapted from https://discuss.pytorch.org/t/how-to-train-with-frozen-batchnorm/12106/8 """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train() if enable else module.eval()

                module.weight.requires_grad = enable
                module.bias.requires_grad = enable

class deepFeatureExtractor_MobileNetV2(nn.Module):
    def __init__(self,args):
        super(deepFeatureExtractor_MobileNetV2, self).__init__()
        self.args = args
        # after passing 1th : H/2  x W/2
        # after passing 2th : H/4  x W/4
        # after passing 3th : H/8  x W/8
        # after passing 4th : H/16 x W/16
        # after passing 5th : H/32 x W/32
        self.encoder = models.mobilenet_v2(pretrained=True)
        del self.encoder.classifier
        self.layerList = [1, 3, 6, 13, 18]
        self.dimList = [16, 24, 32, 96, 960]
        #self.fixList = args.fixlist
    def forward(self, x):
        out_featList = []
        feature = x
        for i in range(len(self.encoder.features)):
            feature = self.encoder.features[i](feature)
            if i in self.layerList:
                out_featList.append(feature)
        return out_featList

    def freeze_bn(self, enable=False):
        """ Adapted from https://discuss.pytorch.org/t/how-to-train-with-frozen-batchnorm/12106/8 """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train() if enable else module.eval()

                module.weight.requires_grad = enable
                module.bias.requires_grad = enable

class deepFeatureExtractor_ResNet101(nn.Module):
    def __init__(self,args, lv6 = False):
        super(deepFeatureExtractor_ResNet101, self).__init__()
        self.args = args
        # after passing ReLU   : H/2  x W/2
        # after passing Layer1 : H/4  x W/4
        # after passing Layer2 : H/8  x W/8
        # after passing Layer3 : H/16 x W/16
        self.encoder = models.resnet101(pretrained=True)
        self.fixList = ['layer1.0','layer1.1','.bn']

        if lv6 is True:
            self.layerList = ['relu','layer1','layer2','layer3', 'layer4']
            self.dimList = [64, 256, 512, 1024,2048]
        else:
            del self.encoder.layer4
            del self.encoder.fc
            self.layerList = ['relu','layer1','layer2','layer3']
            self.dimList = [64, 256, 512, 1024]
        
        for name, parameters in self.encoder.named_parameters():
            if name == 'conv1.weight':
                parameters.requires_grad = False
            if any(x in name for x in self.fixList):
                parameters.requires_grad = False
        
    def forward(self, x):
        out_featList = []
        feature = x
        for k, v in self.encoder._modules.items():
            if k == 'avgpool':
                break
            feature = v(feature)
            #feature = v(features[-1])
            #features.append(feature)
            if any(x in k for x in self.layerList):
                out_featList.append(feature) 
        return out_featList
    def freeze_bn(self, enable=False):
        """ Adapted from https://discuss.pytorch.org/t/how-to-train-with-frozen-batchnorm/12106/8 """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train() if enable else module.eval()

                module.weight.requires_grad = enable
                module.bias.requires_grad = enable

class deepFeatureExtractor_EfficientNet(nn.Module):
    def __init__(self,args, architecture="EfficientNet-B5", lv6 = False):
        super(deepFeatureExtractor_EfficientNet, self).__init__()
        self.args = args
        assert architecture in ["EfficientNet-B0", "EfficientNet-B1", "EfficientNet-B2", "EfficientNet-B3", 
                                    "EfficientNet-B4", "EfficientNet-B5", "EfficientNet-B6", "EfficientNet-B7"]
        
        if architecture == "EfficientNet-B0":
            self.encoder = geffnet.tf_efficientnet_b0_ns(pretrained=True)
            self.dimList = [16, 24, 40, 112, 1280] #5th feature is extracted after conv_head or bn2
            #self.dimList = [16, 24, 40, 112, 320] #5th feature is extracted after blocks[6]
        elif architecture == "EfficientNet-B1":
            self.encoder = geffnet.tf_efficientnet_b1_ns(pretrained=True)
            self.dimList = [16, 24, 40, 112, 1280] #5th feature is extracted after conv_head or bn2
            #self.dimList = [16, 24, 40, 112, 320] #5th feature is extracted after blocks[6]
        elif architecture == "EfficientNet-B2":
            self.encoder = geffnet.tf_efficientnet_b2_ns(pretrained=True)
            self.dimList = [16, 24, 48, 120, 1408] #5th feature is extracted after conv_head or bn2
            #self.dimList = [16, 24, 48, 120, 352] #5th feature is extracted after blocks[6]
        elif architecture == "EfficientNet-B3":
            self.encoder = geffnet.tf_efficientnet_b3_ns(pretrained=True)
            self.dimList = [24, 32, 48, 136, 1536] #5th feature is extracted after conv_head or bn2
            #self.dimList = [24, 32, 48, 136, 384] #5th feature is extracted after blocks[6]
        elif architecture == "EfficientNet-B4":
            self.encoder = geffnet.tf_efficientnet_b4_ns(pretrained=True)
            self.dimList = [24, 32, 56, 160, 1792] #5th feature is extracted after conv_head or bn2
            #self.dimList = [24, 32, 56, 160, 448] #5th feature is extracted after blocks[6]
        elif architecture == "EfficientNet-B5":
            self.encoder = geffnet.tf_efficientnet_b5_ns(pretrained=True)
            self.dimList = [24, 40, 64, 176, 2048] #5th feature is extracted after conv_head or bn2
            #self.dimList = [24, 40, 64, 176, 512] #5th feature is extracted after blocks[6]
        elif architecture == "EfficientNet-B6":
            self.encoder = geffnet.tf_efficientnet_b6_ns(pretrained=True)
            self.dimList = [32, 40, 72, 200, 2304] #5th feature is extracted after conv_head or bn2
            #self.dimList = [32, 40, 72, 200, 576] #5th feature is extracted after blocks[6]
        elif architecture == "EfficientNet-B7":
            self.encoder = geffnet.tf_efficientnet_b7_ns(pretrained=True)
            self.dimList = [32, 48, 80, 224, 2560] #5th feature is extracted after conv_head or bn2
            #self.dimList = [32, 48, 80, 224, 640] #5th feature is extracted after blocks[6]
        if args.local_rank == 0:
            print("==> Model:",architecture)
        del self.encoder.global_pool
        del self.encoder.classifier
        #self.block_idx = [3, 4, 5, 7, 9] #5th feature is extracted after blocks[6]
        #self.block_idx = [3, 4, 5, 7, 10] #5th feature is extracted after conv_head
        self.block_idx = [3, 4, 5, 7, 11] #5th feature is extracted after bn2
        if lv6 is False:
            del self.encoder.blocks[6]
            del self.encoder.conv_head
            del self.encoder.bn2
            del self.encoder.act2
            self.block_idx = self.block_idx[:4]
            self.dimList = self.dimList[:4]
        # after passing blocks[3]    : H/2  x W/2
        # after passing blocks[4]    : H/4  x W/4
        # after passing blocks[5]    : H/8  x W/8
        # after passing blocks[7]    : H/16 x W/16
        # after passing conv_stem    : H/32 x W/32
        self.fixList = ['blocks.0.0','bn']

        for name, parameters in self.encoder.named_parameters():
            if name == 'conv_stem.weight':
                parameters.requires_grad = False
            if any(x in name for x in self.fixList):
                parameters.requires_grad = False
        
    def forward(self, x):
        out_featList = []
        feature = x
        cnt = 0
        block_cnt = 0
        for k, v in self.encoder._modules.items():
            if k == 'act2':
                break
            if k == 'blocks':
                for m, n in v._modules.items():
                    feature = n(feature)
                    if self.block_idx[block_cnt] == cnt:
                        out_featList.append(feature)
                        block_cnt += 1
                    cnt += 1
            else:
                feature = v(feature)
                if self.block_idx[block_cnt] == cnt:
                    out_featList.append(feature)
                    block_cnt += 1
                cnt += 1            
            
        return out_featList

    def freeze_bn(self, enable=False):
        """ Adapted from https://discuss.pytorch.org/t/how-to-train-with-frozen-batchnorm/12106/8 """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train() if enable else module.eval()
                module.weight.requires_grad = enable
                module.bias.requires_grad = enable
