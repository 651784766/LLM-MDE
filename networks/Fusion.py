# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class MultiHeadAttention(nn.Module):
#     def __init__(self, input_dim=512, depth_dim=512, num_heads=8):
#         super(MultiHeadAttention, self).__init__()
#         self.input_dim = input_dim
#         self.depth_dim = depth_dim
#         self.num_heads = num_heads
#         self.attention = nn.MultiheadAttention(embed_dim=self.input_dim, num_heads=self.num_heads)

#     def forward(self, x):
#         out, _ = self.attention(x, x, x)
#         return out + x


# class ResidualAttentionUnit(nn.Module):
#     def __init__(self, features):
#         super().__init__()
#         self.features = features
#         self.attention = MultiHeadAttention(features, num_heads=8)  # 使用 MultiHeadAttention 替代卷积操作
#         self.fc = nn.Linear(features, features)
#         self.relu = nn.ReLU(inplace=False)

#     def forward(self, x):
#         residual = x  # 保存输入作为残差连接的基准
#         out = self.attention(x)  # 使用注意力机制进行特征处理
#         out = self.fc(out)  # 进一步处理特征
#         out = self.relu(out)
#         out += residual  # 将原始输入与处理后的特征相加，实现残差连接
#         return out

# class Fusion(nn.Module):
#     def __init__(self, attention_output2):
#         super(Fusion, self).__init__()
#         self.res_att1 = ResidualAttentionUnit(attention_output2)
#         self.res_att2 = ResidualAttentionUnit(attention_output2)

#     def forward(self, x):
#         previous_stage = torch.zeros_like(x)
#         output_stage1 = self.res_att1(x)
#         output_stage1 += previous_stage
#         output_stage2 = self.res_att2(output_stage1)
#         output_stage2 = nn.functional.interpolate(
#             output_stage2, scale_factor=2, mode="bilinear", align_corners=True
#         )
#         return output_stage2
