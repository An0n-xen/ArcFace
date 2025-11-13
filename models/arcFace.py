import torch
import torch.nn as nn
import torch.nn.functional as F
from losses import ArcFaceLoss


class ArcFaceModel(nn.Module):
    """
    ArcFace model with ResNet50 backbone
    Args:
        num_classes: number of identity classes
        embedding_size: size of embedding features (default 512)
        pretrained: use pretrained ResNet50 weights
    """
    def __init__(self, backbone_model, num_classes, embedding_size=512):
        super().__init__()
        
        # Load ResNet50 backbone
        self.backbone = backbone_model
        
        # Remove the final FC layer
        self.backbone.fc = nn.Identity()
        
        # Feature dimension after ResNet50 (before final FC)
        in_features = 2048
        
        # BN-Dropout-FC-BN structure as described in the paper
        self.bn1 = nn.BatchNorm1d(in_features)
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(in_features, embedding_size)
        self.bn2 = nn.BatchNorm1d(embedding_size)
        
        # ArcFace head
        self.arcface = ArcFaceLoss(embedding_size, num_classes, s=64.0, m=0.5)
        
    def forward(self, x, labels=None):
        # Extract features with ResNet50
        x = self.backbone(x)
        
        # BN-Dropout-FC-BN
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.fc(x)
        embedding = self.bn2(x)
        
        if labels is not None:
            # Training mode: return ArcFace logits
            output = self.arcface(embedding, labels)
            return output, embedding
        else:
            # Inference mode: return normalized embeddings
            return F.normalize(embedding)