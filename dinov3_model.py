from torch import nn
import timm


class DINOv3(nn.Module):
    def __init__(self, freeze_backbone=True, pool='token', model_name='vit_large_patch16_dinov3.lvd1689m'):
        super().__init__()

        print(f"Initializing {model_name}...")

        # Load Backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,      # No classification head
            global_pool=pool
        )

        # Freeze logic
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("Backbone Frozen.")


    def forward(self, x):
        # x: [B, 3, 224, 224]
        feats = self.backbone(x)
        return feats  # [B, 1024]