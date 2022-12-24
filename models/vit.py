from timm.models import vision_transformer
import torch.nn as nn


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=32, pretrained=True, num_classes=10):
        super(VisionTransformer, self).__init__()
        model_kwargs = dict(
            patch_size=patch_size, img_size=img_size, embed_dim=384, depth=12, num_heads=6
        )
        self.vit = vision_transformer._create_vision_transformer(
            "vit_small_patch32_224", pretrained=pretrained, **model_kwargs
        )
        if num_classes != 1000:
            self.vit.head = nn.Linear(self.vit.embed_dim, num_classes)
            self.vit.num_classes = num_classes

    def forward(self, x):
        return self.vit(x)


if __name__ == "__main__":
    vit = VisionTransformer()
