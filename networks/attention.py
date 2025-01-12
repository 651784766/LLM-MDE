
import torch.nn as nn
import torch
import torch.nn.functional as F

# 需不需要softmax或者激活函数？


class MultiheadAttentionLayer(nn.Module):
    def __init__(self, input_dim=512, depth_dim=512, num_heads=8):
        super(MultiheadAttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.depth_dim = depth_dim
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(embed_dim=self.input_dim, num_heads=self.num_heads)

    def forward(self, HR_tensor, depth_tensor, image_features):
        attention_output, _ = self.attention(depth_tensor, image_features, HR_tensor)
        return attention_output



class MultiheadAttentionLayer2(nn.Module):
    def __init__(self, input_dim=512, depth_dim=512, num_heads=8):
        super(MultiheadAttentionLayer2, self).__init__()
        self.input_dim = input_dim
        self.depth_dim = depth_dim
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(embed_dim=self.input_dim, num_heads=self.num_heads)

    def forward(self, attention_output, text_features):
        attention_output2, _ = self.attention(attention_output, attention_output, text_features)
        return attention_output2

