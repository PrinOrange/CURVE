import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel


# SupCon 损失函数
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode="all"):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode

    def forward(self, features, labels=None):
        device = features.device
        bsz = features.shape[0]

        # L2 normalize
        features = F.normalize(features, dim=-1)

        # 合并视角 -> [N * V, dim]
        contrast_features = torch.cat(torch.unbind(features, dim=1), dim=0)

        # 计算相似度矩阵
        similarity = torch.div(
            torch.matmul(contrast_features, contrast_features.T), self.temperature
        )

        # mask 去除自己
        logits_mask = torch.ones_like(similarity).fill_diagonal_(0)

        if labels is None:
            # 无监督对比学习
            mask = torch.eye(bsz, dtype=torch.float32, device=device)
        else:
            # supervised contrast: same-class are positive pairs
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(device)

        # 扩展 mask -> 对应多视角
        mask = mask.repeat(features.shape[1], features.shape[1])

        # 去掉自己
        mask = mask * logits_mask

        # log-softmax
        exp_logits = torch.exp(similarity) * logits_mask
        log_prob = similarity - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        # positive log_prob: 只计算正例
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-12)

        # loss
        loss = -mean_log_prob_pos.mean()

        return loss


class RobertaSupCon(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained(model_name)
        self.projector = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 128),
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]  # CLS embedding
        z = self.projector(cls)
        return z
