import torch
import torch.nn as nn
import torch.nn.functional as F

class InterpretationConsistencyLoss(nn.Module):
    """
    L_int = 1 - sim(R_prev, R_curr)
    强制相邻层级的 CWE 语义表示对齐
    """
    def __init__(self, similarity="cosine"):
        super().__init__()
        
        if similarity == "cosine":
            self.sim_fn = self._cosine_sim
        elif similarity == "dot":
            self.sim_fn = self._dot_sim
        else:
            raise ValueError("similarity must be 'cosine' or 'dot'")

    def _cosine_sim(self, x, y):
        return F.cosine_similarity(x, y, dim=-1)  # [B]

    def _dot_sim(self, x, y):
        return torch.sum(x * y, dim=-1)  # [B]

    def forward(self, R_prev, R_curr):
        """
        R_prev: 上一层的表示 [B, D]
        R_curr: 当前层表示   [B, D]
        """
        sim = self.sim_fn(R_prev, R_curr)   # [B]
        loss = 1 - sim                      # [B]
        return loss.mean()
