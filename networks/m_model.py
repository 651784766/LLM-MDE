import torch.nn as nn
from networks.attention import MultiheadAttentionLayer ,MultiheadAttentionLayer2
import timm
import torch
import networks.HRNet_pre as HRNet_pre
import networks.lite_mono_depth as lite_mono_depth
import json
import torch.nn.functional as F
from tools.Deconv import Deconv1,Deconv2
from transformers import CLIPProcessor, CLIPModel
import tunning.layers as layers


class Model_Cus(nn.Module):

    def __init__(self):
        super(Model_Cus, self).__init__()
    
        with open('config.json', 'r') as config_file:
            config = json.load(config_file)
            # config.json
        self.device = config["General"]["device"]
        self.batch_size = config["General"]["batch_size"]
        self.supervised =  config["General"]["supervised"] 
        self.resize = config["Dataset"]["transforms"]["resize"]

        # 模态缺失
        self.Missing_modality = config["General"]["Missing_modality"]  

        # invariant
        self.HR_tensor = HRNet_pre
        self.lite_mono_depth =lite_mono_depth

        # variant
        self.depth_map_fc = nn.Linear(512,192 * 640)
        self.depth_map_fc2 =  nn.Linear(512,480 * 640)

        # paramenter init
        self.attention_layer_image = MultiheadAttentionLayer().to(self.device)
        self.attention_layer_image2 = MultiheadAttentionLayer2().to(self.device)

        # 1,192,640 -> 1,8,24
        self.conv_4 = nn.Conv2d(1, 64, kernel_size=3, stride=3, padding=1).to(self.device)
        self.conv_5 = nn.Conv2d(64, 64, kernel_size=3, stride=3, padding=1).to(self.device)
        self.conv_6 = nn.Conv2d(64, 1, kernel_size=3, stride=3, padding=1).to(self.device)

        #vit_attention
        #self.vit_model = timm.create_model('vit_base_patch16_384', pretrained=True)  # 加载预训练权重

        from transformers import AutoImageProcessor, AutoModelForImageClassification
        self.vit_model  = AutoModelForImageClassification.from_pretrained("./model/VIT").to(self.device)
        self.vit_processor = AutoImageProcessor.from_pretrained("./model/VIT")
        self.clip_model1 = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(self.device)
        self.processor1 = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        
        self.linear_layer = torch.nn.Linear(768, 512)


        #需要固定初始化矩阵大小
        self.mlp_2 = nn.Linear(8*24, 512).to(self.device)
        self.depth_map_fc = nn.Linear(512, 480 * 640).to(self.device)


        self.fc_1 = nn.Linear(1000, 512).to(self.device)
        self.fc_2 = nn.Linear(1000, 512).to(self.device)

        self.fc_3 = nn.Linear(1000, 512).to(self.device)


        # 实例化 upsample
        self.deconv_layer1 = Deconv1(supervised=self.supervised,resize=self.resize) 
        self.deconv_layer2 = Deconv2() 
        # 所有参数
        self.parameters_all = [{'params': [param for  param in self.named_parameters()]}]

        # 将所有参数设为可训练
        for name, param in self.named_parameters():
            #param.requires_grad = True  

            #print(name, param.requires_grad)  #vit包含在内且全部为true
            if 'clip' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True


        #检测
        #for name, param in self.vit_model.named_parameters():
        #    if not param.requires_grad:
        #        print(f"Parameter {name} is frozen.")
        #    else:
        #        print(f"Parameter {name} is trainable.")


        # 这是linear层的输出q_new = x * Wq


        # for param in self.vit_model.parameters():
        #     param.requires_grad = True  
        
        # 让低秩可训练
        #for name, param in self.named_parameters():
        #    if 'lora' in name:
        #        param.requires_grad = True

        
        # for i in range(12):
        #     self.vit_model.blocks[i].attn.qkv = layers.Linear(768, 2304, r=2)

        #self.vit_model.blocks[2].attn.qkv =  layers.Linear(768, 2304, r=2)
                
        for i,block in enumerate(self.vit_model.vit.encoder.layer):
            block.attention.attention.query = layers.Linear(768, 768, r=2)
            block.attention.attention.key = layers.Linear(768, 768, r=2)
            block.attention.attention.value = layers.Linear(768, 768, r=2)

        #for param in self.vit_model.parameters():
        #    param.requires_grad = True

        #for name, param in self.vit_model.named_parameters():
        #    if "blocks.7.attn.qkv.weight" in name:

        #        print(param.data)
        
        #print(param.data.shape)  [2304,768] = [input_feature,output_feature]

     
     # x1 x2 x3 -> y
    def forward(self,pixel_values, text_features, encoded_text_line2, sup_tensor):

        #print(pixel_values.shape)  #bs,3,384,384
        #print(text_features.shape)  #bs,1,512
        #print(encoded_text_line2.shape)  #bs,512
        #print(sup_tensor.shape)  #bs,1,384,384


        # 图像clip  bs,512
        inputs_image = self.processor1(images=pixel_values, return_tensors="pt").to(self.device)
        clip_image = self.clip_model1.get_image_features(**inputs_image)

        #print(clip_image.shape)
        #print('encoded_text_line2',encoded_text_line2.shape)


        # bs,512  
        encoded_text_line2 

        # bs,512 -> bs,1024
        clip_image_text = torch.cat([clip_image, encoded_text_line2], dim=1)

        # upsample  bs,1024 -> bs,3,384,384
        clip_image_text = self.deconv_layer2(clip_image_text)

        # 根据 resize 参数调整输出尺寸
        #clip_image_text = nn.functional.interpolate(clip_image_text, size=(self.resize, self.resize), mode='bilinear')

        clip_image_text=clip_image_text.to(self.device)

        # bs,3,384,384 -> bs,1000
        clip_image_text = (clip_image_text - clip_image_text.min()) / (clip_image_text.max() - clip_image_text.min()).to(self.device)

        inputs = self.vit_processor(images=clip_image_text, return_tensors="pt").to(self.device)
        outputs = self.vit_model(**inputs)
        # bs,1000
        vit_image = outputs.logits

        # bs,1000 -> bs,512
        image_features = self.fc_3(vit_image)

        # bs,1000
        HR_tensor = self.HR_tensor.hrnet_inference(pixel_values)

        # bs,1000  -> bs,512
        HR_tensor = self.fc_1(HR_tensor)

        # litemono bs,1,192,640
        depth_tensor = self.lite_mono_depth.test_simple(pixel_values)

        # bs,1,192,640 -> bs,1,8,24
        conv_output_2 = self.conv_6(self.conv_5(self.conv_4(depth_tensor)))

        # bs,8,24
        conv_output_2 = conv_output_2.squeeze(1)

        # bs,8*24
        conv_output_2 = conv_output_2.view(conv_output_2.size(0),-1)

        # bs,8*24 @ 8*24,512 -> bs,512
        lite_depth_map = self.mlp_2(conv_output_2)

        # bs,512 # 
        attention_output = self.attention_layer_image(depth_tensor=lite_depth_map,image_features=image_features,HR_tensor=HR_tensor)

        # bs,1,512 -> bs,512
        text_features = text_features.squeeze(1)

        # 嵌入了文本2的图像路和单独的clip文本路 attention
        attention_output2 = self.attention_layer_image2(attention_output=attention_output, text_features=text_features)

        # upsample
        depth_map,depth_tensor = self.deconv_layer1(attention_output2, sup_tensor)

        # depth_map是预测  depth_tensor是标签或者lite output
        return depth_map ,depth_tensor
