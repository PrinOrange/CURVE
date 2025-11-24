import torch
import torch.nn as nn
from transformers import RobertaConfig, RobertaModel, RobertaPreTrainedModel

class RobertaForSequenceClassification(RobertaPreTrainedModel):
    def __init__(self, config: RobertaConfig):
        super().__init__(config)
        self.num_labels = config.num_labels  # 多分类时用
        self.roberta = RobertaModel(config)
        hidden_size = config.hidden_size

        # ------- Attention Pooling -------
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1)
        )

        # -------- Classifier ---------
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(hidden_size, config.num_labels)

        # init weights
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.roberta(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )

        last_hidden_state = outputs.last_hidden_state  # (B, L, H)

        # attention pooling
        attn_scores = self.attention(last_hidden_state)  # (B, L, 1)
        attn_scores = attn_scores.masked_fill(
            attention_mask.unsqueeze(-1) == 0, float("-inf")
        )
        attn_weights = torch.softmax(attn_scores, dim=1)
        pooled_output = torch.sum(attn_weights * last_hidden_state, dim=1)

        # classifier
        logits = self.classifier(self.dropout(pooled_output))

        loss = None
        if labels is not None:
            if self.num_labels == 1:  # binary regression-like
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits.view(-1), labels.float())
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels)

        return {
            "loss": loss,
            "logits": logits,
        }
