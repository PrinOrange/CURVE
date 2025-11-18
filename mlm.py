import os
import torch
from datasets import load_dataset
from Consts.Paths import MODELS_DIR
from transformers import (
    RobertaTokenizer,
    RobertaConfig,
    RobertaForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# -----------------------------
# 配置
# -----------------------------
HF_DATASET_PATH = "codemetic/curve"
HF_DATASET_SUBSET = "pretrain"
HF_DATASET_SPLIT = "train[:100]"
OUTPUT_DIR = "./out"

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# 加载数据集
# -----------------------------
corpus = load_dataset(
    path=HF_DATASET_PATH,
    name=HF_DATASET_SUBSET,
    split=HF_DATASET_SPLIT,
    streaming=False  # 训练需要知道长度
)

# -----------------------------
# 初始化 tokenizer
# GraphCodeBERT 基于 Roberta
# -----------------------------
tokenizer = RobertaTokenizer.from_pretrained(MODELS_DIR)

# -----------------------------
# Tokenize dataset
# -----------------------------
def preprocess(examples):
    # 对 source 列 tokenize
    return tokenizer(
        examples["source"],
        truncation=True,
        max_length=512
    )

tokenized_corpus = corpus.map(
    preprocess,
    batched=True,
    remove_columns=["source", "id"]
)

print(tokenized_corpus[0])

# -----------------------------
# Data collator for MLM
# -----------------------------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

# -----------------------------
# 模型配置
# -----------------------------
config = RobertaConfig.from_pretrained("microsoft/graphcodebert-base")
model = RobertaForMaskedLM(config=config).to(device)

print("Model parameters:", model.num_parameters())

# -----------------------------
# Training arguments
# -----------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    do_train=True,
    per_device_train_batch_size=8,
    num_train_epochs=10,
    save_steps=10000,
    save_total_limit=4,
    logging_steps=500,
    seed=42,
)

# -----------------------------
# Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_corpus,
    data_collator=data_collator,
)

# -----------------------------
# 开始训练
# -----------------------------
trainer.train()
