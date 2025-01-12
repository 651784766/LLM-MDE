import torch
from torchvision import transforms
from PIL import Image
from networks.cls_hrnet import get_cls_net
#from text_image_dataset  import MyDataset
#from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm  
import os
import json



def hrnet_inference(pixel_values):
    # 设置GPU
    torch.cuda.set_device(0)  # 选择要使用的GPU设备
    # 创建数据集实例时传递 config 参数



    # HRNet模型初始化
    model = get_cls_net({
        'GPUS': (0,),
        'MODEL': {
            'NAME': 'cls_hrnet',
            'IMAGE_SIZE': [224, 224],
            'EXTRA': {
                'STAGE1': {'NUM_MODULES': 1, 'NUM_RANCHES': 1, 'BLOCK': 'BOTTLENECK', 'NUM_BLOCKS': [4], 'NUM_CHANNELS': [64], 'FUSE_METHOD': 'SUM'},
                'STAGE2': {'NUM_MODULES': 1, 'NUM_BRANCHES': 2, 'BLOCK': 'BASIC', 'NUM_BLOCKS': [4, 4], 'NUM_CHANNELS': [18, 36], 'FUSE_METHOD': 'SUM'},
                'STAGE3': {'NUM_MODULES': 4, 'NUM_BRANCHES': 3, 'BLOCK': 'BASIC', 'NUM_BLOCKS': [4, 4, 4], 'NUM_CHANNELS': [18, 36, 72], 'FUSE_METHOD': 'SUM'},
                'STAGE4': {'NUM_MODULES': 3, 'NUM_BRANCHES': 4, 'BLOCK': 'BASIC', 'NUM_BLOCKS': [4, 4, 4, 4], 'NUM_CHANNELS': [18, 36, 72, 144], 'FUSE_METHOD': 'SUM'}
            }
        }
    })

    model = torch.nn.DataParallel(model, device_ids=[0]).cuda()

    # 检查点路径
    checkpoint_path = './model/hrnetv2_w18_imagenet_pretrained.pth'  # 替换为你的模型检查点路径

    # 加载模型权重
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict, strict=False)  # 使用 strict=False 忽略不匹配的层

    # 数据预处理
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    val_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        normalize,
    ])

    pixel_values = val_transform(pixel_values)#.unsqueeze(0) 

    # 数据加载
    #input_image = val_transform(Image.open(image_path).convert('RGB')).unsqueeze(0)

    # 模型推理，获取全连接层之前的输出
    with torch.no_grad():
        model.eval()
        output = model(pixel_values)
        
    HR_tensor = output  # 根据实际情况调整
    # 在这里你可以处理 HR_tensor，或者将它们存储到列表中

    return HR_tensor



#hrnet_tensors = hrnet_inference()
