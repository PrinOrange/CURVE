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
    EarlyStoppingCallback,
)

# -----------------------------
# 配置
# -----------------------------
RANDOM_SEED = 42
EPOCHS = 1
BATCH_SIZE = 64

CHECK_STEPS = 2000
LOGGING_STEPS = 500
EARLY_STOP_PATIENT = 5

HF_DATASET_PATH = "codemetic/curve"
HF_DATASET_SUBSET = "pretrain"
HF_DATASET_SPLIT = "train"
HF_MODEL_PATH = "microsoft/graphcodebert-base"
TESTSET_SIZE = 0.05

MLM_PROBABILITY = 0.15
MAX_TOKEN_LENGTH = 512

OUTPUT_DIR = "./out"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# -----------------------------
# 加载数据集
# -----------------------------
raw_corpus = load_dataset(
    path=HF_DATASET_PATH,
    name=HF_DATASET_SUBSET,
    split=HF_DATASET_SPLIT,
)

# -----------------------------
# 划分 train / eval
# -----------------------------
split = raw_corpus.train_test_split(
    test_size=TESTSET_SIZE, seed=RANDOM_SEED
)  # 通常 1%～5% 都够用
train_corpus = split["train"]
eval_corpus = split["test"]

print("Train size:", len(train_corpus))
print("Eval size:", len(eval_corpus))

# -----------------------------
# 初始化 tokenizer
# -----------------------------
tokenizer = RobertaTokenizer.from_pretrained(MODELS_DIR)


# -----------------------------
# Tokenization
# -----------------------------
def preprocess(examples):
    return tokenizer(
        examples["source"],
        truncation=True,
        max_length=MAX_TOKEN_LENGTH,
    )


tokenized_train = train_corpus.map(
    preprocess, batched=True, remove_columns=["source", "id"]
)

tokenized_eval = eval_corpus.map(
    preprocess, batched=True, remove_columns=["source", "id"]
)

# -----------------------------
# MLM collator
# -----------------------------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=MLM_PROBABILITY,
)

# -----------------------------
# 模型
# -----------------------------
config = RobertaConfig.from_pretrained(HF_MODEL_PATH)
model = RobertaForMaskedLM.from_pretrained(HF_MODEL_PATH).to(DEVICE)

print("Model parameters:", model.num_parameters())

# -----------------------------
# Training args（含早停机制）
# -----------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    do_train=True,
    # ==== 训练批次 ====
    per_device_train_batch_size=BATCH_SIZE,
    # ==== 训练轮数 ====
    num_train_epochs=EPOCHS,
    # ==== 评估 ====
    eval_strategy="steps",
    eval_steps=CHECK_STEPS,  # 每 2000 steps 评估一次
    load_best_model_at_end=True,  # 自动加载最佳 checkpoint
    metric_for_best_model="loss",  # 使用 eval_loss 选择最佳模型
    greater_is_better=False,  # loss 越低越好
    # ==== Checkpoint ====
    save_steps=CHECK_STEPS,
    save_strategy="steps",
    save_total_limit=4,
    # ==== Logging ====
    logging_steps=LOGGING_STEPS,
    # ==== Early stopping ====
    # 若 5 次 eval 后 eval_loss 无改善 → 停止训练
    # patience = eval 次数，不是 epoch
    seed=RANDOM_SEED,
)

# -----------------------------
# Trainer + EarlyStopping
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    data_collator=data_collator,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PATIENT)  # 早停
    ],
)

# -----------------------------
# 开始训练
# -----------------------------
trainer.train()
