from transformers import RobertaModel
import torch.nn as nn

class CweBERT(nn.Module):
    """
    RoBERTa-Based 模型，作为特征提取器和分类器头。
    """
    def __init__(self, model_name, num_cwe):
        super().__init__()
        # 特征提取器 E(X)
        self.encoder = RobertaModel.from_pretrained(model_name)
        # 降维投影头 (可选，用于对比学习)
        self.proj_head = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, self.encoder.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.encoder.config.hidden_size, 128) # 128是对比学习的嵌入维度
        )
        # CWE 分类头 (用于硬/软标签损失)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_cwe)

    def forward(self, input_ids, attention_mask):
        # 获取 [CLS] token 的表示 R
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]
        
        # 1. 嵌入表示 Z (用于 SupCon 损失)
        z = self.proj_head(cls_token)
        
        # 2. Logits (用于硬/软标签损失)
        logits = self.classifier(cls_token)
        
        # 3. 内部特征 R_int (用于 L_int 损失)
        r_int = cls_token # 使用 cls_token 作为内部表示

        return z, logits, r_int