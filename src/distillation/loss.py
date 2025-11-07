# Feature + Mask distillation loss
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.7, temperature=4.0):
        super().__init__()
        self.alpha = alpha
        self.T = temperature
        self.mse = nn.MSELoss()
        self.kld = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_feat, teacher_feat, student_mask, teacher_mask):
        # Feature alignment (L2 on normalized features)
        s_feat = F.normalize(student_feat, dim=1)
        t_feat = F.normalize(teacher_feat, dim=1)
        feat_loss = self.mse(s_feat, t_feat)

        # Mask logit distillation (soft labels)
        s_logit = student_mask / self.T
        t_logit = teacher_mask / self.T
        mask_loss = self.kld(
            F.log_softmax(s_logit, dim=1),
            F.softmax(t_logit, dim=1)
        ) * (self.T ** 2)

        return self.alpha * feat_loss + (1 - self.alpha) * mask_loss