import torch
from depth import clip

def zeroshot_classifier(depth_classes,obj_classes, templates, model):
    with torch.no_grad():
        zeroshot_weights = []
        for depth in depth_classes:
            for obj in obj_classes:
                texts = [template.format(obj,depth) for template in templates]  # format with class
                texts = clip.tokenize(texts).cuda()  # tokenize
                class_embeddings = model.encode_text(texts)  # embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights