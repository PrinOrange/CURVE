import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftLabelLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.T = temperature
        self.kl = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, teacher_logits):
        """
        student_logits: [B, C]
        teacher_logits: [B, C]
        """
        T = self.T
        # 1. 学生分布（log_softmax）
        student_log_prob = F.log_softmax(student_logits / T, dim=-1)
        # 2. 教师分布（softmax）
        teacher_prob = F.softmax(teacher_logits / T, dim=-1)
        # 3. KL(student || teacher)
        loss = self.kl(student_log_prob, teacher_prob) * (T * T)

        return loss
