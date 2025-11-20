import torch
from datasets import load_dataset
from transformers import (
    RobertaTokenizer,
    RobertaForMaskedLM,
    DataCollatorForLanguageModeling,
)
import json
from datetime import datetime

# ----------------------------
# 配置
# ----------------------------
HF_MODEL_PATH = "/home/malaoshi/bak/checkpoint-42000"
HF_DATASET_PATH = "codemetic/curve"
HF_DATASET_SUBSET = "pretrain"
HF_DATASET_SPILT = "train"

MAX_TOKEN_LENGTH = 512
BATCH_SIZE = 64
MLM_PROBABILITY = 0.15

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# 加载模型和 tokenizer
# ----------------------------
tokenizer = RobertaTokenizer.from_pretrained(HF_MODEL_PATH)
model = RobertaForMaskedLM.from_pretrained(HF_MODEL_PATH)
model.eval()

# ----------------------------
# 加载数据集
# ----------------------------
raw_ds = load_dataset(HF_DATASET_PATH, HF_DATASET_SUBSET, split=HF_DATASET_SPILT)


# -----------------------------
# Tokenization
# -----------------------------
def preprocess(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_TOKEN_LENGTH,
    )


tokenized_ds = raw_ds.map(preprocess, batched=True, remove_columns=["text"])

# ----------------------------
# MLM 数据增强（随机 mask）
# ----------------------------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=MLM_PROBABILITY
)

# ----------------------------
# DataLoader
# ----------------------------
from torch.utils.data import DataLoader

dataloader = DataLoader(
    tokenized_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=data_collator
)

# ----------------------------
# 评估
# ----------------------------
model.to(DEVICE)

total_loss = 0
count = 0

# Accuracy 统计变量
masked_correct = 0
top5_correct = 0
top10_correct = 0
total_masked_tokens = 0

with torch.no_grad():
    for batch in dataloader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

        loss = outputs.loss
        total_loss += loss.item()
        count += 1

        logits = outputs.logits  # [B, L, vocab]
        labels = batch["labels"]  # [B, L]
        mask_positions = labels != -100  # 找出被mask的位置

        # 如果某个 batch 完全没有 mask（极少概率），跳过
        if mask_positions.sum() == 0:
            continue

        # 取 mask 位置的 logits 和 labels
        masked_logits = logits[mask_positions]  # [M, vocab]
        masked_labels = labels[mask_positions]  # [M]
        total_masked_tokens += masked_labels.size(0)

        # -----------------------------
        # Top-1（Masked Token Accuracy）
        # -----------------------------
        pred_top1 = masked_logits.argmax(dim=-1)
        masked_correct += (pred_top1 == masked_labels).sum().item()

        # -----------------------------
        # Top-5 Accuracy
        # -----------------------------
        top5 = masked_logits.topk(5, dim=-1).indices
        top5_correct += (top5 == masked_labels.unsqueeze(-1)).any(dim=1).sum().item()

        # -----------------------------
        # Top-10 Accuracy
        # -----------------------------
        top10 = masked_logits.topk(10, dim=-1).indices
        top10_correct += (top10 == masked_labels.unsqueeze(-1)).any(dim=1).sum().item()


# ----------------------------
# 最终指标
# ----------------------------
avg_loss = total_loss / count
perplexity = torch.exp(torch.tensor(avg_loss))

masked_acc = masked_correct / total_masked_tokens
top5_acc = top5_correct / total_masked_tokens
top10_acc = top10_correct / total_masked_tokens

print(f"MLM Test Loss           = {avg_loss:.4f}")
print(f"Perplexity              = {perplexity:.4f}")
print(f"Masked Token Accuracy   = {masked_acc:.4%}")
print(f"Top-5 Accuracy          = {top5_acc:.4%}")
print(f"Top-10 Accuracy         = {top10_acc:.4%}")

# ============================================================
# 将结果写入 JSON 文件（以数组形式保存多个实验）
# ============================================================
result = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model_name": HF_MODEL_PATH,
    "dataset": HF_DATASET_SUBSET,
    "split": HF_DATASET_SPILT,
    "max_length": MAX_TOKEN_LENGTH,
    "batch_size": BATCH_SIZE,
    "mlm_probability": MLM_PROBABILITY,
    "test_loss": float(avg_loss),
    "perplexity": float(perplexity),
    "masked_accuracy": float(masked_acc),
    "top5_accuracy": float(top5_acc),
    "top10_accuracy": float(top10_acc),
}

json_file = "mlm-evaluation.report.json"

# 如果文件存在 → 读取 → append → 再写入
try:
    with open(json_file, "r") as f:
        data = json.load(f)
except FileNotFoundError:
    data = []

data.append(result)

with open(json_file, "w") as f:
    json.dump(data, f, indent=2)

print(f"\n实验结果已写入 {json_file}")
