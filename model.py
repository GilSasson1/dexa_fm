import torch.nn as nn
import timm
import torch

class LeJEPA_Encoder(nn.Module):
    def __init__(self, model_name='resnet50', img_size=(224, 224), proj_out_dim=256, pretrained=False, drop_path_rate=0.0):
        super().__init__()

        # ViTs usually use the [CLS] token.
        # CNNs (ResNet, ConvNeXt, EfficientNet) use Average Pooling.
        is_vit = 'vit' in model_name or 'swin' in model_name

        model_kwargs = {
            "pretrained": pretrained,
            "num_classes": 0,
            "drop_path_rate": drop_path_rate,
        }

        if is_vit:
            model_kwargs['img_size'] = img_size
            model_kwargs["dynamic_img_size"] = True
            model_kwargs['global_pool'] = 'token' # Use [CLS] token for ViTs
        else:
            # Crucial for ResNet/ConvNeXt: Force Average Pooling
            # Otherwise you get (B, C, H, W) which crashes the Linear layer below
            model_kwargs['global_pool'] = 'avg'

            # Create Backbone
        self.backbone = timm.create_model(model_name, **model_kwargs)
        self.embed_dim = self.backbone.num_features

        # Standard design: input_dim -> 2048 -> 2048 -> proj_out_dim
        self.projector = nn.Sequential(
            nn.Linear(self.embed_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, proj_out_dim),
        )

    def forward(self, x):
        # Features will now always be (Batch, embed_dim) regardless of arch
        features = self.backbone(x)
        projections = self.projector(features)
        return features, projections

class SIGReg(nn.Module):
    def __init__(self, num_slices=2048, knots=17, integration_limit=5):
        super().__init__()
        self.num_slices = num_slices

        # OPTIMIZATION: Integrate only [0, limit] instead of [-limit, limit]
        # This doubles the resolution (dt is smaller) for the same cost.
        t = torch.linspace(0, integration_limit, knots, dtype=torch.float32)
        
        # Calculate dt for the half-range
        dt = integration_limit / (knots - 1)
        
        # Standard Trapezoidal Weights for [0, limit]
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt

        target_cf = torch.exp(-0.5 * t.square())

        self.register_buffer("t", t.view(1, 1, -1))
        self.register_buffer("weights", weights)
        self.register_buffer("target_cf", target_cf.view(1, -1))

    def forward(self, z):
        B, D = z.shape 

        # Projections
        A = torch.randn(D, self.num_slices, device=z.device)
        A = A / (A.norm(dim=0, keepdim=True) + 1e-6)  # Normalize each column to unit length
        z_proj = z @ A 

        # ECF
        val = z_proj.unsqueeze(-1) * self.t
        ecf_real = val.cos().mean(dim=0)
        ecf_imag = val.sin().mean(dim=0)

        # Error
        diff = (ecf_real - self.target_cf).square() + ecf_imag.square()
        diff = diff * self.target_cf 

        # Integrate
        # Because we baked the "* 2.0" into the weights, this result 
        # automatically represents the full [-5, 5] integral.
        loss_per_slice = diff @ self.weights 

        return loss_per_slice.mean() * B