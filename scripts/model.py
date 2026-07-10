import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        x = x.mean(dim=1, keepdim=True).pow(1.0 / self.p)
        return x.squeeze(1)
    
class EventViTStudent(nn.Module):
    def __init__(self,
                 backbone_name='vit_small_patch16_dinov3.lvd1689m',
                 teacher_dim=1024,
                 num_patches=576,
                 img_size=(480, 640),
                 in_channels=3):
        super().__init__()

        self.teacher_dim = teacher_dim
        self.num_patches = num_patches
        self.image_size = img_size
        self.in_channels = in_channels

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,
            img_size=self.image_size
        )

        student_dim = self.backbone.embed_dim
        print(f"Student backbone dimension: {student_dim}")
        print(f"Teacher dimension: {teacher_dim}")

        self.input_norm = nn.BatchNorm2d(in_channels)

        self.proj = nn.Linear(student_dim, teacher_dim)
        self.gem = GeM()

    def forward(self, x):
        x = self.input_norm(x)
        
        features = self.backbone.forward_features(x)
        patches = features[:, -self.num_patches:, :]
        projected_patches = self.proj(patches)
        global_descriptor = self.gem(projected_patches)
        
        return projected_patches, global_descriptor