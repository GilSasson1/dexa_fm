import torch.nn as nn
import timm

class LeJEPA_Encoder(nn.Module):
    def __init__(self, model_name='resnet50', proj_out_dim=256, pretrained=False, drop_path_rate=0.0):
        super().__init__()

        # 1. Logic to distinguish ViTs from CNNs
        # ViTs usually use the [CLS] token.
        # CNNs (ResNet, ConvNeXt, EfficientNet) use Average Pooling.
        is_vit = 'vit' in model_name or 'swin' in model_name

        model_kwargs = {
            "pretrained": pretrained,
            "num_classes": 0,
            "drop_path_rate": drop_path_rate  # <--- Essential for your run_sweep.sh!
        }

        if is_vit:
            model_kwargs["dynamic_img_size"] = True
            model_kwargs['global_pool'] = 'token' # Use [CLS] token for ViTs
        else:
            # Crucial for ResNet/ConvNeXt: Force Average Pooling
            # Otherwise you get (B, C, H, W) which crashes the Linear layer below
            model_kwargs['global_pool'] = 'avg'

            # 2. Create Backbone
        self.backbone = timm.create_model(model_name, **model_kwargs)
        self.embed_dim = self.backbone.num_features

        # 3. Projector (The "Expander")
        # Standard design: input_dim -> 2048 -> 2048 -> proj_out_dim
        self.projector = nn.Sequential(
            nn.Linear(self.embed_dim, 2048, bias=False),
            nn.ReLU(),
            nn.Linear(2048, 2048, bias=False),
            nn.ReLU(),
            nn.Linear(2048, proj_out_dim, bias=False),
        )

    def forward(self, x):
        # Features will now always be (Batch, embed_dim) regardless of arch
        features = self.backbone(x)
        projections = self.projector(features)
        return features, projections