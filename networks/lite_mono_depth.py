from __future__ import absolute_import, division, print_function
import os
import torch
#import networks
from PIL import ImageFile

#from models import  networks
from modules.depth_decoder import DepthDecoder

from modules.depth_encoder import LiteMono
from torchvision import transforms
ImageFile.LOAD_TRUNCATED_IMAGES = True



def test_simple(pixel_values):

    pixel_values = pixel_values.to('cuda:0')
    encoder_path = os.path.join('./model', "encoder.pth")
    decoder_path = os.path.join('./model', "depth.pth")

    encoder_dict = torch.load(encoder_path)
    decoder_dict = torch.load(decoder_path)


    encoder = LiteMono(model="lite-mono",
                                height=192,
                                width=640)

    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})

    encoder.to('cuda:0')
    encoder.eval()

    depth_decoder = DepthDecoder(encoder.num_ch_enc, scales=range(3))
    depth_model_dict = depth_decoder.state_dict()
    depth_decoder.load_state_dict({k: v for k, v in decoder_dict.items() if k in depth_model_dict})

    depth_decoder.to('cuda:0')
    depth_decoder.eval()

    with torch.no_grad():
        
        augmentation1 = transforms.Resize((192, 640))
        pixel_values_lite = augmentation1(pixel_values)#.unsqueeze(0)

        features = encoder(pixel_values_lite)
        outputs = depth_decoder(features)

        disp = outputs[("disp", 0)]

        #print(disp.shape)
        depth_tensor= disp
        #1,1,192,640   深度图
        return depth_tensor




# 随机初始化一个1x3x192x640的张量
random_tensor = torch.randn(1, 3, 192, 640)

image_path= './input/299.jpg'
test_simple(random_tensor)

