from networks.m_model  import Model_Cus
import torch
import torch.nn as nn
from tqdm import tqdm
import json
import os
import numpy as np
#from loss import ssim
from pytorch_msssim import ssim

def validate(epoch, device, val_data_loader, num_epochs):

    model_cus = Model_Cus().to(device)
    Mse = nn.MSELoss()
    val_losses = []
    val_losses_per_epoch = []
    average_val_loss= []


    count=1
    model_cus.eval()
    with torch.no_grad():

        for batch in tqdm(val_data_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Validating  ' ,ncols=150):
        

            pixel_values, text_features, encoded_text_line2, sup_tensor = batch


            text_features = text_features.to(device)
            encoded_text_line2 = encoded_text_line2.to(device)
            pixel_values  =  pixel_values.to(device)
            sup_tensor = sup_tensor.to(device)


            depth_map , depth_tensor = model_cus(pixel_values, text_features, encoded_text_line2, sup_tensor)


            # X,1,192,640
            # X,1,480,640

            loss_ssim =  ssim(depth_map, depth_tensor, data_range=1)
            loss_mse = Mse(depth_map, depth_tensor)
            loss= loss_mse 

            val_losses.append(loss.item())
            #print('val_losses:',val_losses)
            count +=1

        #每个epoch平均损失
        average_val_loss = sum(val_losses) / len(val_losses)

        val_losses_per_epoch.append(average_val_loss)



    return average_val_loss


