import torch
import torch.nn as nn
import torch.nn.functional as F

class HardLabelLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, student_logits, hard_labels):
        """
        student_logits: [B, C]
        hard_labels:    [B]
        """
        return self.ce(student_logits, hard_labels)
