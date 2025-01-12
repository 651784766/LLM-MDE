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


def evalution(device,test_data_loader):


    model_cus = Model_Cus().to(device)

    checkpoint = torch.load('./weight/test_model.pth')
    model_cus.load_state_dict(checkpoint['model_state_dict'])
    model_cus = model_cus

    test_metrics_list = []
    model_cus.eval()

    with torch.no_grad():

        for batch in tqdm(test_data_loader):
        

            pixel_values, text_features, encoded_text_line2, sup_tensor = batch


            text_features = text_features.to(device)
            encoded_text_line2 = encoded_text_line2.to(device)
            #HR_tensor = HR_tensor.to(device)
            #depth_tensor = depth_tensor.to(device)
            pixel_values  =  pixel_values.to(device)
            sup_tensor = sup_tensor.to(device)


            depth_map , depth_tensor = model_cus(pixel_values, text_features, encoded_text_line2, sup_tensor)


            test_metrics = calculate_metrics(depth_map,depth_tensor)

            test_metrics_list.append(test_metrics)

    

        average_metrics = {}
        num_samples = len(test_metrics_list)

        for metric_name in test_metrics_list[0].keys():
            metric_sum = sum(metrics[metric_name] for metrics in test_metrics_list)
            average_metrics[metric_name] = metric_sum / num_samples

        return average_metrics
    

    
import os

# plt
def test(device):

    model_cus = Model_Cus().to(device)

    checkpoint = torch.load('./weight/test_model.pth')
    model_cus.load_state_dict(checkpoint['model_state_dict'])
    model_cus = model_cus


    model_cus.eval()

    with torch.no_grad():

        for batch in tqdm(eval_data_loader):
        
            pixel_values, text_features, encoded_text_line2, sup_tensor = batch

            text_features = text_features.to(device)
            encoded_text_line2 = encoded_text_line2.to(device)
            pixel_values  =  pixel_values.to(device)
            sup_tensor = sup_tensor.to(device)


            depth_map = model_cus(pixel_values, text_features, encoded_text_line2, sup_tensor)

            os.makedirs("./output", exist_ok=True)
            # 1,1,480,640 -> 1,480,640 for: F.to_pil_image

            for i, single_depth_map in enumerate(depth_map):

                save_path = f"./output/{i + 1}.jpg"

                single_depth_map = single_depth_map.squeeze(0)

                image = TF.to_pil_image(single_depth_map)

                image.save(save_path)







