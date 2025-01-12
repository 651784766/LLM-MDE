import torch
import torch.nn as nn
import torch.nn.functional as F
from depth.layers.classifier_layer import zeroshot_classifier
from depth.layers.fc_layer import FCLayer
from depth.models.lora import LoRA_ViT_timm
from depth import clip
import timm
from math import sqrt
from transformers import BertConfig, BertModel, BertTokenizer, GPT2Model, GPT2Config, GPT2Tokenizer,LlamaConfig,LlamaTokenizer, LlamaForCausalLM
    
import loralib as lora


class Language4Depth(nn.Module):


    def __init__(self, configs, prompt=False, LoRA=False):
        super(Language4Depth, self).__init__()

        self.lora = LoRA
        self.vit_backbone = timm.create_model('vit_base_patch16_224', pretrained=True)

        if self.lora:
            self.vit_lora = LoRA_ViT_timm(vit_model=self.vit_backbone, r=configs.rank, alpha=configs.alpha, num_classes=10) # vit frozen + low-rank params. vit-based encoder

            # # Set requires_grad appropriately for specific qkv layers in selected blocks
            # for name, param in self.vit_lora.named_parameters():
            #     if any(f"blocks.{i}.attn.qkv" in name for i in [2,5,8,11]):
            #         param.requires_grad = True
            #     else:
            #         param.requires_grad = False
        else:
            self.vit_lora = self.vit_backbone
            for param in self.vit_lora.parameters():
                param.requires_grad = False  # 冻结参数

        if configs.llm_model == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')
            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )

            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False
                )

        elif configs.llm_model == 'BERT':

            self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')
            self.bert_config.num_hidden_layers = configs.llm_layers
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True

            try:

                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )

            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False
                )


        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        for param in self.llm_model.parameters():
            param.requires_grad = False

        if configs.prompt_domain:
            self.descripttion = configs.content
        else:
            self.description = 'Monocular depth estimation is the process of inferring the distance of objects in a scene from a single 2D image.'
        
        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        self.ada_pool = nn.AdaptiveMaxPool2d((224, 224))

        self.d_ff = configs.d_ff #768 default

        if self.lora:
            self.d_model = self.vit_lora.lora_vit.embed_dim
        else:
            self.d_model = 768
        #self.d_model = self.vit_lora.lora_vit.embed_dim

        self.d_llm = self.llm_model.get_input_embeddings().weight.shape[1]
        self.cross_att = CrossAttention(self.d_model, configs.n_heads, self.d_model, self.d_llm)

        # fixed modules fit all dataset
        self.l2d_module = Language2Depth(self.d_model, reduction=4)
        self.l2d_deocder = Language2DepthDecoder(3)
        self.mapping_layer2 = nn.Linear(196, 3)
        self.mapping_layer3 = nn.Linear(self.d_model, 1024)
        self.mapping_layer4 = nn.Linear(310, 3) 
        self.mapping_layer5 = nn.Linear(self.d_model, 1024)
        self.prompt = prompt

        self.depth_es = nn.Sigmoid()
        # self.bert_config1 = BertConfig.from_pretrained('google-bert/bert-base-uncased')
        # self.bert_config1.num_hidden_layers = 12
        #LoRA for LLM
        self.llm_decoder = BertModel.from_pretrained(
            'google-bert/bert-base-uncased',
            #config=self.bert_config,
        )
        # for param in self.llm_decoder.parameters():
        #     param.requires_grad = False
            
        if configs.Microsoft_LoRA != 0:
            for layer in self.llm_decoder.encoder.layer:
                layer.attention.self.query = lora.Linear(768, 768, r=configs.Microsoft_LoRA)
                layer.attention.self.key = lora.Linear(768, 768, r=configs.Microsoft_LoRA)
                layer.attention.self.value = lora.Linear(768, 768, r=configs.Microsoft_LoRA)

            # Set requires_grad appropriately
            for name, param in self.llm_decoder.named_parameters():
                if "encoder.layer" in name and ("query" in name or "key" in name or "value" in name):
                    param.requires_grad = True
                else:
                    param.requires_grad = False


    def forward(self, x):
        orig_x = x
        #       x (bs, 480, 640) ---->  x fit vit's shape (bs, 224, 224)
        # x = self.ada_pool(x)
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        if self.lora:
            enc_out = self.vit_lora.lora_vit.forward_features(x)[:, :-1]
        else:
            enc_out = self.vit_lora.forward_features(x)[:, :-1]

        # Prototype
        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
        enc_out = self.cross_att(enc_out, source_embeddings, source_embeddings)

        #       [text prototype & image feature] ----> language to depth ----> l2d feature (bs, 192, 768)
        # linear relu linear relu
        enc_out_l2d = self.l2d_module(enc_out)  # bs,192,768

        #       if have prompt in this model
        if self.prompt:
            max_values_x, min_values_x, median_values_x, \
                max_values_y, min_values_y, median_values_y = self.stat_compute(x)
            prompt = []
            for b in range(x.shape[0]):
                #print(type(min_values_x[b]), min_values_x[b]) # <class 'int'> 223
                #print(type(min_values_y[b]), min_values_y[b])

                min_value_str_xaxis = str(min_values_x[b]) # 233
                # min_value_str_xaxis = str(min_values_x[b].tolist()[0])  
                min_value_str_yaxis = str(min_values_y[b])

                max_value_str_xaxis = str(max_values_x[b])
                max_value_str_yaxis = str(max_values_y[b])

                median_values_str_xaxis = str(median_values_x[b])
                median_values_str_yaxis = str(median_values_y[b])

                depth_str = str(self.depth_compute(x))

                
                prompt_ = (
                    f"<|start_prompt|>Dataset description: {self.description}"
                    f"Task description: estimate the photo's depth"
                    "Input statistics:"
                    f"min value on the x axis {min_value_str_xaxis},"
                    f"min value on the y axis {min_value_str_yaxis},"
                    f"max value on the x axis {max_value_str_xaxis},"
                    f"max value on the y axis {max_value_str_yaxis},"
                    f"median value on the x axis {median_values_str_xaxis},"
                    f"median value on the y axis {median_values_str_yaxis},"
                    f"the depth class of input is {depth_str}<|<end_prompt>|>"
                )
                # CO-STAR Framework 1.C-context  2.O-objective 3.S-Style  4.T-Tone  5.A-audience 6.R-Response 

                # prompt_ = (
                #     f"<|start_prompt|>Dataset description: NYU-v2 Dataset."
                #     f"Context: Monocular depth estimation is the process of inferring the distance of objects in a scene from a single 2D image."
                #     f"Objective: Accurately estimate the pixel values of each point in the image as depth information and output the depth map."
                #     f"Style: Use a rigorous scientific style."
                #     f"Audience: Computer or deep learning models."
                #     f"Response: Tensor or binary text<|<end_prompt>|>"
                # )

                prompt.append(prompt_)

            prompt = self.tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=2048).input_ids
            prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x.device))  #(bs, prompt_token, dim)

            enc_out = torch.cat([prompt_embeddings, enc_out_l2d], dim=1) # bs,310,768

            dec_out = self.llm_decoder(inputs_embeds=enc_out).last_hidden_state
            
            dec_out = dec_out[:, :, :self.d_ff] #(bs, prompt_token + 196, 768) if True, mapping_layer need updated.   # 16,312,768 
            dec_out = dec_out[:, :310, :]

            #dec_out = enc_out[:, :310, :] 

            # enc_out_l2d ----> upsample -----> depth estimation (bs, 480, 640)
            dec_out = self.mapping_layer4(dec_out.permute(0, 2, 1)).permute(0, 2, 1)  # bs,3,768 

            dec_out = self.mapping_layer5(dec_out).reshape(-1, 3, 32, 32) # (bs, 3, 32, 32)
            # upsample ,include conv and relu 
            dec_out = self.l2d_deocder(dec_out, orig_x) # 16,1,480,640  

        else:

            # enc_out_l2d ----> upsample -----> depth estimation (bs, 480, 640)
            enc_out_l2d = self.mapping_layer2(enc_out_l2d.permute(0, 2, 1)).permute(0, 2, 1)  # bs,3,768

            enc_out_l2d = self.mapping_layer3(enc_out_l2d).reshape(-1, 3, 32, 32) # (bs, 3, 32, 32)

            dec_out = self.l2d_deocder(enc_out_l2d, orig_x) # 16,1,480,640 

        #dec_out = F.interpolate(x, size=[dec_out.shape[2], dec_out.shape[3]], mode='bilinear', align_corners=True)

        depth = self.depth_es(dec_out) #sigmoid  

        return depth
    
    def depth_compute(self, x):
        depth_classes = ['giant', 'extremely close', 'close', 'not in distance', 
                         'a little remote', 'far', 'unseen']
        bin_list = [1.00, 1.50, 2.00, 2.25, 2.50, 2.75, 3.00]

        bin_list = torch.tensor(bin_list).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        # Assume the depth information in the 1st channel
        depth_channel = x[0]

        depth_values, counts = torch.unique(depth_channel, return_counts=True)
        most_frequent_depth = depth_values[torch.argmax(counts)]

        class_index = torch.bucketize(most_frequent_depth.detach().clone(), torch.tensor(bin_list))

        if most_frequent_depth > bin_list[-1]:
            class_index = len(bin_list)

        return depth_classes[class_index.item()]

    
    def stat_compute(self, x):
        assert len(x.shape) == 4
        bs, channels, height, width = x.shape

        _, max_y_indices = torch.max(x, dim=2)
        _, min_y_indices = torch.min(x, dim=2)
        median_y_indices = torch.median(x, dim=2).indices

        _, max_x_indices = torch.max(x, dim=3)
        _, min_x_indices = torch.min(x, dim=3)
        median_x_indices = torch.median(x, dim=3).indices

        max_x_list = []
        min_x_list = []
        median_x_list = []
        max_y_list = []
        min_y_list = []
        median_y_list = []
        
        for i in range(bs):
            max_x = max_x_indices[i].min().item()
            min_x = min_x_indices[i].max().item()
            median_x = torch.median(median_x_indices[i]).item()

            max_y = max_y_indices[i].min().item()
            min_y = min_y_indices[i].max().item()
            median_y = torch.median(median_y_indices[i]).item()
            
            max_x_list.append(max_x)
            min_x_list.append(min_x)
            median_x_list.append(median_x)
            max_y_list.append(max_y)
            min_y_list.append(min_y)
            median_y_list.append(median_y)
        
        return max_x_list, min_x_list, median_x_list, \
                max_y_list, min_y_list, median_y_list

class CrossAttention(nn.Module):
    """
        CorssAttention for multimodal feature alignment and fusion
        args1 - d_model
        args2 - n_head: number of attention head
        args3 - d_keys: as the same as d_model
        args4 - d_llm: the dimension of pre-trained LLM
        args5 - attention_dropout: dropout rate
        ----------------------------------------------------------
        input1 - target_embedding: feature maps from vit encoding
        input2 - source_embeeding: pre-trained word embeeding from pre-trained LLM
        input3 - source_embeeding: pre-trained word embeeding from pre-trained LLM
    
    """
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(CrossAttention, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, D = target_embedding.shape
        S, _ = source_embedding.shape

        #print('n_heads=',self.n_heads)  
        H = self.n_heads


        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)# Q是image提供的，包含batchsize，而KV是大模型生成的，没有包含B
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, source_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)
        scores = torch.einsum('blhe,she->bhls', target_embedding, source_embedding)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls, she->blhe", A, value_embedding)
        return reprogramming_embedding

class Language2Depth(nn.Module):
    def __init__(self, c_in, reduction):
        super(Language2Depth, self).__init__()

        self.l2p_l1 = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

        self.l2p_l2 = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        return x + self.l2p_l2(self.l2p_l1(x))
        

# BN feature=4, d_model=3
class Language2DepthDecoder(nn.Module):
    def __init__(self, d_model) -> None:
        super(Language2DepthDecoder, self).__init__()

        features = int(d_model)
        
        self.up1 = UpSampleBN(skip_input=2 * features, output_features=4 * features)

        self.up2 = UpSampleBN(skip_input=5 * features, output_features=8 * features)
        self.up3 = UpSampleBN(skip_input=9 * features, output_features=12 * features)
        self.conv3 = nn.Conv2d(12 * features, 1, kernel_size=3, stride=1, padding=1)
        # self.act_out = nn.Softmax(dim=1) if output_activation == 'softmax' else nn.Identity()

    def forward(self, enc_out, x):
        x_d1 = self.up1(enc_out, x) 
        x_d2 = self.up2(enc_out, x_d1) 
        x_d3 = self.up3(enc_out, x_d2)
        out = self.conv3(x_d3) 
        return out

# con+RELU+con+RELU
class UpSampleBN(nn.Module):
    def __init__(self, skip_input, output_features):
        super(UpSampleBN, self).__init__()

        self._net = nn.Sequential(nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU())
    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.shape[2], concat_with.shape[3]], mode='bilinear', align_corners=True)
        f = torch.cat([up_x, concat_with], dim=1)
        return self._net(f)





if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    depth_map = torch.rand(size=(2, 3, 480, 640)).cuda()

    model = Language2Depth(224, 224, 768).cuda()

    output = model(depth_map)

    print(model(depth_map).shape)

    # depth_map = torch.rand(size=(2, 3, 196, 768)).cuda()
    # contact_map = torch.rand(size=(2, 3, 480, 640)).cuda()

    # uplayer = UpSampleBN(skip_input=6, output_features=640).cuda()

    # print(uplayer(depth_map, contact_map).shape)

    # output = output.cpu().detach().numpy()

    # encc_out = torch.rand(size=(2, 196, 768)).cuda()
    # source_embedding = torch.rand(size=(1000, 768)).cuda()

    # layer = CorssAttention(768, 4, 768, 768).cuda()

    # opt = layer(encc_out, source_embedding, source_embedding)
    # print(opt.shape)





