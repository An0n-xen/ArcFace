import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ArcFaceLoss(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss (Core Implementation)

    Paper: ArcFace: Additive Angular Margin Loss for Deep Face Recognition
    Formula (Eq. 3): L = -log(e^(s*cos(θ_yi + m)) / (e^(s*cos(θ_yi + m)) + Σ e^(s*cos(θ_j))))

    Args:
        in_features: embedding dimension (e.g., 512)
        out_features: number of classes
        s: feature scale (default 64.0)
        m: angular margin in radians (default 0.5)
    """
    def __init__(self, in_features, out_features, s=64.0, m=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m

        # Weight matrix: [num_classes, embedding_dim]
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        # Pre-compute constants for cos(θ + m) computation
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)  # threshold to prevent θ + m > π
        self.mm = math.sin(math.pi - m) * m  # for numerical stability

    def forward(self, input, label):
        """
        Args:
            input: embeddings [batch_size, in_features]
            label: ground truth labels [batch_size]

        Returns:
            logits: [batch_size, out_features]
        """
        # 1. Normalize input features and weights (L2 norm)
        cosine = F.linear(F.normalize(input), F.normalize(self.weight.to(input.device)))

        # 2. Calculate sin(θ) from cos(θ)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))

        # 3. Calculate cos(θ + m) using: cos(θ+m) = cos(θ)cos(m) - sin(θ)sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m

        # 4. Numerical stability: if cos(θ) < cos(π-m), use alternative formula
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # 5. Apply margin only to target class (one-hot)
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # 6. Combine: cos(θ+m) for target, cos(θ) for others
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        # 7. Scale by s
        output *= self.s

        return output    
    

class ArcFaceLayer(nn.Module):
    def __init__(self, num_classes, arc_m=0.5, arc_s=64.0, regularizer_l=5e-4):
        """
        ArcFace layer implemented in PyTorch.

        Args:
            num_classes (int): Number of classes.
            arc_m (float): Additive angular margin (default=0.5).
            arc_s (float): Feature scale factor (default=64.0).
            regularizer_l (float): L2 regularization weight (default=5e-4).
        """
        super().__init__()
        self.num_classes = num_classes
        self.arc_m = arc_m
        self.arc_s = arc_s
        self.regularizer_l = regularizer_l

        self.cos_m = math.cos(arc_m)
        self.sin_m = math.sin(arc_m)
        self.th = math.cos(math.pi - arc_m)
        self.mm = math.sin(math.pi - arc_m) * arc_m

        # Weight parameter (512 should match your embedding dimension)
        self.weight = nn.Parameter(torch.empty(512, num_classes))
        nn.init.xavier_normal_(self.weight)

    def forward(self, features, labels):
        # Normalize features and weights
        embedding_norm = F.normalize(features, p=2, dim=1)
        weights_norm = F.normalize(self.weight, p=2, dim=0)

        # Cosine similarity between features and weights
        cos_t = torch.matmul(embedding_norm, weights_norm)
        cos_t = cos_t.clamp(-1, 1)  # numerical stability

        # Compute sin_t and cos(mt)
        sin_t = torch.sqrt(1.0 - torch.pow(cos_t, 2))
        cos_mt = self.arc_s * (cos_t * self.cos_m - sin_t * self.sin_m)

        # Threshold condition
        cond = cos_t - self.th
        keep_val = self.arc_s * (cos_t - self.mm)
        cos_mt_temp = torch.where(cond > 0, cos_mt, keep_val)

        # One-hot encode labels
        mask = F.one_hot(labels, num_classes=self.num_classes).float()
        inv_mask = 1.0 - mask

        # Combine logits
        s_cos_t = self.arc_s * cos_t
        output = s_cos_t * inv_mask + cos_mt_temp * mask

        return output
