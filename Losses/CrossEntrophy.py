import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLossModule(nn.Module):
    def __init__(self, weight=None, reduction="mean", from_logits=True):
        """
        封装交叉熵损失

        Args:
            weight (Tensor, optional): 各类别的权重，形状 [num_classes]
            reduction (str): 'mean' / 'sum' / 'none'
            from_logits (bool): 如果 True，输入 logits，否则输入概率分布
        """
        super().__init__()
        self.weight = weight
        self.reduction = reduction
        self.from_logits = from_logits

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (batch_size, num_classes) logits 或概率
            targets: (batch_size,) 长整型标签，或 (batch_size, num_classes) one-hot
        Returns:
            loss: 标量或 (batch_size,) 根据 reduction
        """
        if self.from_logits:
            # PyTorch 的 cross_entropy 接收 logits
            if targets.dim() == 2:
                # one-hot -> class index
                targets = targets.argmax(dim=1)
            loss = F.cross_entropy(
                inputs, targets, weight=self.weight, reduction=self.reduction
            )
        else:
            # 输入是概率分布
            if targets.dim() == 1:
                # 长整型标签 -> one-hot
                num_classes = inputs.size(1)
                targets = F.one_hot(targets, num_classes=num_classes).float()
            loss = -(targets * torch.log(inputs + 1e-12)).sum(dim=1)  # per sample
            if self.reduction == "mean":
                loss = loss.mean()
            elif self.reduction == "sum":
                loss = loss.sum()
            # else 'none' -> keep per-sample loss
        return loss
