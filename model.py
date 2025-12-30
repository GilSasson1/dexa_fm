import torch.nn as nn
import timm
import torch

class LeJEPA_Encoder(nn.Module):
    def __init__(self, model_name='resnet50', proj_out_dim=256, pretrained=False, drop_path_rate=0.0):
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
            model_kwargs["dynamic_img_size"] = True
            model_kwargs['global_pool'] = 'token' # Use [CLS] token for ViTs
        else:
            # Crucial for ResNet/ConvNeXt: Force Average Pooling
            # Otherwise you get (B, C, H, W) which crashes the Linear layer below
            model_kwargs['global_pool'] = 'avg'

            # Create Backbone
        self.backbone = timm.create_model(model_name, **model_kwargs)
        self.embed_dim = self.backbone.num_features

        # Projector (The "Expander")
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

class SIGReg(nn.Module):
    def __init__(self, num_slices=2048, knots=17, integration_limit=5):
        super().__init__()
        self.num_slices = num_slices

        # Pre-calculate integration points (The "Grid")
        t = torch.linspace(-integration_limit, integration_limit, knots, dtype=torch.float32)
        dt = (2 * integration_limit) / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt

        # Gaussian target function (The "Ideal Shape")
        target_cf = torch.exp(-0.5 * t.square())

        # Register as buffers so they move to GPU automatically but aren't trained
        self.register_buffer("t", t.view(1, 1, -1)) # Shape for broadcasting
        self.register_buffer("weights", weights)
        self.register_buffer("target_cf", target_cf.view(1, -1))

    def forward(self, z):
        """
        z: (Batch_Size * Views, Dim)
        """
        B, D = z.shape

        # Generate Random Projections (The "Slices")
        A = torch.randn(D, self.num_slices, device=z.device)
        A = A.div_(A.norm(dim=0, keepdim=True) + 1e-6)

        # Project Data -> (Batch, Slices)
        z_proj = z @ A

        # Compute Empirical Characteristic Function (ECF)
        # val: (Batch, Slices, Knots)
        val = z_proj.unsqueeze(-1) * self.t

        # ecf: (Slices, Knots) -> Average over Batch
        ecf_real = val.cos().mean(dim=0)
        ecf_imag = val.sin().mean(dim=0)

        # Compute Weighted Error
        diff = (ecf_real - self.target_cf).square() + ecf_imag.square()

        # Integrate error (Trapezoidal rule)
        loss_per_slice = diff @ self.weights

        # Return Mean over slices
        return loss_per_slice.mean()