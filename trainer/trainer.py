from networks.m_model  import Model_Cus
import torch
from tqdm import tqdm
import numpy as np
from pytorch_msssim import ssim
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import ExponentialLR


def train(epoch,device,num_epochs,train_data_loader,opt_cus_all):

    model_cus = Model_Cus().to(device)
    Mse = nn.MSELoss()

    # params
    # total_params = 0
    # for param in model_cus.parameters():
    #     if param.requires_grad:
    #         num_elements = param.numel()
    #         total_params += num_elements

    # print(f"Total params: {total_params}")



    train_losses = []
    average_train_loss= []

    model_cus.train()
    count = 1



    for batch in tqdm(train_data_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training  ' ,ncols=150):
    

        pixel_values, text_features, encoded_text_line2, sup_tensor = batch


        text_features = text_features.to(device)
        encoded_text_line2 = encoded_text_line2.to(device)
        pixel_values  =  pixel_values.to(device)
        sup_tensor = sup_tensor.to(device)


        depth_map , depth_tensor = model_cus(pixel_values, text_features, encoded_text_line2, sup_tensor)



        loss_ssim =ssim(depth_map, depth_tensor, data_range=1)
        loss_mse = Mse(input=depth_map, target=depth_tensor)
        loss =  0.5* loss_ssim + 0.5*loss_mse

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model_cus.parameters(), max_norm= 1) 
        opt_cus_all.step()
        # 更新学习率



        # #检测梯度是否更新 包括vit都没问题
        # for name, param in model_cus.named_parameters():
        #     if param.grad is not None:
        #         print(name)
        #         print(param.grad.shape)
        #     else:
        #         print(f"No gradient for parameter {name}")


        # #检测权重是否更新
        # for name, param in model_cus.vit_model.named_parameters():
        #     if "blocks.8.attn.qkv.weight" in name:

        #         print(param.data)


        opt_cus_all.zero_grad()

        train_losses.append(loss.item())
        count +=1

    average_train_loss = sum(train_losses) / len(train_losses)


       
    return average_train_loss
