from networks.m_model  import Model_Cus
import torch
from tqdm import tqdm
from tools.tools import calculate_metrics
from reader import MyDataset
import json
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader


with open('config.json', 'r') as config_file:
    config = json.load(config_file)
type =  config["Dataset"]["type"]


eval_dataset = MyDataset(f'./dataset/data_{type}.csv', config, split_type="eval")
eval_data_loader = DataLoader(eval_dataset, batch_size=1)


    
import os

# plt
def test(device):

    model_cus = Model_Cus().to(device)
    #加载模型
    checkpoint = torch.load('./weight/test_model.pth')
    model_cus.load_state_dict(checkpoint['model_state_dict'])

    model_cus.eval()

    with torch.no_grad():

        for batch in tqdm(eval_data_loader):
        
            pixel_values, text_features, encoded_text_line2 = batch

            text_features = text_features.to(device)
            encoded_text_line2 = encoded_text_line2.to(device)
            pixel_values  =  pixel_values.to(device)

            sup_tensor = torch.rand(224, 224).to(device)
            

            depth_map = model_cus(pixel_values, text_features, encoded_text_line2,sup_tensor)

            os.makedirs("./output", exist_ok=True)
            # 1,1,480,640 -> 1,480,640 for: F.to_pil_image

            for i, single_depth_map in enumerate(depth_map):

                save_path = f"./output/{i + 1}.jpg"

                single_depth_map = single_depth_map.squeeze(0)
                # 将张量转换为 PIL.Image 对象
                image = TF.to_pil_image(single_depth_map)

                # 保存图像
                image.save(save_path)


test(device='cuda')


