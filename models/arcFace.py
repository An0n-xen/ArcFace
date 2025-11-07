import torch
from torch import nn


class ArchFaceHead(nn.Module):
    def __init__(self, in_features=512, num_classess=1000, s=64.0, m=0.5):
        super().__init__()

        self.weight = nn.Parameter(torch.FloatTensor(num_classess, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m

    def forward(self, input_features, labels):
        # Compute logits with additive angular margin
        normalized_feat = nn.functional.normalize(input_features, p=2, dim=1)
        normalized_weight = nn.functional.normalize(self.weight, p=2, dim=1)

        # Cosine similarity
        cosine = torch.matmul(normalized_feat, normalized_weight.t())
        # select the cos value for the correct class
        theta = torch.acos(
            cosine[torch.arange(len(labels)), labels].clamp(-1 + 1e-7, 1 - 1e-7)
        )

        # add margin
        target_logit = torch.cos(theta + self.m)

        logits = cosine.clone()
        logits[torch.arange(len(labels)), labels] = target_logit

        logits = logits * self.s
        return logits


class ArcFaceModel(nn.Module):
    def __init__(self, num_classes, backbone):
        super().__init__()

        self.backbone = backbone
        self.fc = ArchFaceHead(in_features=512, num_classess=num_classes)

    def forward(self, images, labels=None):
        features = self.backbone(images)

        if labels is not None:
            logits = self.fc(features, labels)
            return logits, features
        else:
            return features
