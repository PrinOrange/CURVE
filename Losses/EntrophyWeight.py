import torch
import torch.nn as nn
import torch.nn.functional as F

def ClassComplementaryCoefficient(rho: float):
    if rho <= 50:
        return 0.99
    elif rho <= 500:
        return 0.999
    else:
        return 0.9999

class EntropyWeight(nn.Module):
    """
    根据教师模型概率分布计算归一化熵，
    以及基于熵的硬标签动态权重 u = 1 - e。
    """

    def __init__(self, num_classes:float,rho:float, eps=1e-12):
        super().__init__()
        self.num_classes = num_classes
        self.rho = rho
        self.eps = eps
        self.log_num_classes = torch.log(torch.tensor(float(num_classes)))

    def forward(self, teacher_probs):
        """
        teacher_probs: (batch, num_classes)，必须是 softmax 后的概率
        """
        # ---- 防止 log(0) ----
        probs = teacher_probs.clamp(min=self.eps)
        # ---- 计算负 log 概率 ----
        log_probs = torch.log(probs)
        # ---- 交叉熵项 -∑ p log p ----
        entropy = -(probs * log_probs).sum(dim=-1)  # (batch,)
        # ---- 归一化熵 ----
        norm_entropy = entropy / self.log_num_classes  # (batch,)
        # ---- 动态硬标签权重 ----
        u = 1 - norm_entropy  # (batch,)
        return norm_entropy, u
