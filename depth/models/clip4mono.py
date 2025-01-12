# Hyperparameter Control:
depth_templates = ['This {} is {}'] 
obj_classes=['object']
depth_classes =['giant', 'extremely close', 'close','not in distance','a little remote', 'far','unseen'] 
bin_list=[1.00, 1.50, 2.00, 2.25, 2.50, 2.75, 3.00]
temperature=0.1
clip_vis = 'RN50'

from depth.clip import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from depth.layers.classifier_layer import zeroshot_classifier
from depth.layers.fc_layer import FCLayer


class MonoCLIP(nn.Module):
    def __init__(self):
        super(MonoCLIP, self).__init__()
        self.bins = len(depth_classes)

        self.clip, _ = clip.load(clip_vis) # load pretrained clip encoder
        self.text_f = zeroshot_classifier(depth_classes, obj_classes, depth_templates, self.clip) # init text feature

        self.adapter = FCLayer(1024).to(self.clip.dtype)

    def forward(self, x):
        # clip's image_encode receives input (bs, 3, 256, 256)
        img_f = self.clip.encode_image(x).permute(1, 0, 2)  # B, HW, C
        img_f = img_f / img_f.norm(dim=-1, keepdim=True) # normalize img_f

        # @: dot product of two vectors
        img_f = torch.nn.functional.interpolate(img_f,scale_factor=0.5) # to match size

        depth_logits = 100. * img_f @ self.text_f  # B, HW, K # img_f and text_f have both been normalized, so just use a inner product
        depth_logits = depth_logits.permute(0, 2, 1).reshape(-1, self.bins, 13, 17)  # B, K, H, W 
        depth_logits /= temperature

        depth = F.softmax(depth_logits, dim=1)
        bin_tensor=torch.tensor(bin_list).to(depth.device)
        depth = depth * bin_tensor.reshape(1, self.bins).unsqueeze(-1).unsqueeze(-1)
        depth = depth.sum(1, keepdim=True)
        return depth   

if __name__ == '__main__':
    depth_map = torch.rand(size=(2, 224, 224, 3)).cuda()

    model = MonoCLIP().cuda()

    print(model(depth_map).shape)


