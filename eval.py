import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    matthews_corrcoef,
    confusion_matrix,
)
from collections import Counter
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel

# ========================
# 配置：请根据你的训练设置修改以下路径和参数
# ========================

# 替换为你的实际输出目录路径（包含 best_model.pth）
OUTPUT_DIR = "output_primevul-paired_20251211-181517"  # 👈 修改这里！

# 数据集配置（应与训练时一致）
DATASET_NAME = "codemetic/curve"
SUBSET_NAME = "primevul-paired"
MODEL_NAME = "codemetic/CweBERT-mlm"
MAX_LENGTH = 152
BATCH_SIZE = 64
ROBERTA_LAYERS_TO_CONCAT = (6, 7, 8, 9)

# 评估哪个 split？可选: "test", "val", "train"
EVAL_SPLIT = "test"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ========================
# 重用你的模型定义（简化版，仅用于加载）
# ========================


class KappaFaceHead(torch.nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(num_classes, in_features))
        torch.nn.init.xavier_uniform_(self.weight)
        # 注意：没有 bias，与训练一致

    def forward(self, x):
        # 评估时不用，随便写
        return x


class RoBERTaEncoder(torch.nn.Module):
    def __init__(self, model_name, layers_to_concat=(6, 7, 8, 9)):
        super().__init__()
        self.config = RobertaConfig.from_pretrained(
            model_name, output_hidden_states=True
        )
        self.roberta = RobertaModel.from_pretrained(model_name, config=self.config)
        self.layers_to_concat = layers_to_concat
        self.hidden_size = self.config.hidden_size
        self.concat_dim = len(layers_to_concat) * self.hidden_size

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states
        selected_layers = []
        for layer_idx in self.layers_to_concat:
            actual_idx = -(layer_idx)  # e.g., -6 → index 7 in 0-based 13-length list
            selected_layers.append(hidden_states[actual_idx])
        concatenated = torch.cat(selected_layers, dim=-1)
        cls_embedding = concatenated[:, 0, :]
        return cls_embedding


class VulnerabilityModel(torch.nn.Module):
    def __init__(self, model_name, num_classes, layers_to_concat=(6, 7, 8, 9)):
        super().__init__()
        self.encoder = RoBERTaEncoder(model_name, layers_to_concat)
        self.feature_dim = self.encoder.concat_dim
        # 关键：使用 KappaFaceHead 模块，而不是直接 Parameter
        self.kappaface_head = KappaFaceHead(self.feature_dim, num_classes)

    def forward(self, input_ids, attention_mask):
        features = self.encoder(input_ids, attention_mask)
        return torch.nn.functional.normalize(features, p=2, dim=-1)


# ========================
# 数据集类（简化）
# ========================


class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        source = str(item["source"])
        label = bool(item["label"])
        cwe = str(item["cwe"])
        class_key = (label, cwe)
        inputs = self.tokenizer(
            source,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "label": label,
            "cwe": cwe,
            "class_key": class_key,
        }


# ========================
# 辅助函数
# ========================


def load_class_mappings(output_dir):
    with open(os.path.join(output_dir, "class_mappings.json"), "r") as f:
        data = json.load(f)
    class_to_idx = {}
    for k, v in data["class_to_idx"].items():
        # k is like "True|CWE-123"
        parts = k.split("|", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid class key format: {k}")
        label_str, cwe = parts
        label = label_str == "True"  # boolean from string
        class_to_idx[(label, cwe)] = v
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return class_to_idx, idx_to_class


def compute_thresholds(train_embeddings, train_labels, prototypes, idx_to_class):
    """
    为每个类别计算阈值 τ，基于训练集内该类样本的相似度分布。
    """
    thresholds = {}
    train_embs_np = train_embeddings.cpu().numpy()
    train_labels_np = train_labels.cpu().numpy()

    for c in range(len(idx_to_class)):
        mask = train_labels_np == c
        if not mask.any():
            thresholds[c] = -1.0  # impossible to accept
            continue

        class_embs = torch.tensor(train_embs_np[mask], device=DEVICE)
        sims = torch.mm(class_embs, prototypes[c].unsqueeze(0).T).squeeze(1)  # (N,)
        sims = sims.cpu().numpy()

        best_f1, best_tau = 0.0, 0.5
        for tau in np.linspace(0.0, 1.0, 100):
            preds = (sims >= tau).astype(int)
            if preds.sum() == 0:
                continue
            # 真实标签全为1（因为是该类样本）
            f1 = precision_recall_fscore_support(
                np.ones_like(preds), preds, average="binary", zero_division=0
            )[2]
            if f1 > best_f1:
                best_f1 = f1
                best_tau = tau
        thresholds[c] = best_tau
    return thresholds

def custom_collate_fn(batch):
    """
    Custom collate function to prevent stacking of non-tensor fields like class_key, cwe, label.
    """
    elem = batch[0]
    result = {}
    for key in elem:
        if key in ["class_key", "cwe", "label"]:
            # Keep as list of Python objects
            result[key] = [d[key] for d in batch]
        else:
            # Stack tensors (input_ids, attention_mask)
            result[key] = torch.stack([d[key] for d in batch])
    return result


# ========================
# 主函数
# ========================


def main():
    print(f"Loading model from: {OUTPUT_DIR}")
    checkpoint_path = os.path.join(OUTPUT_DIR, "best_model.pth")
    assert os.path.exists(checkpoint_path), f"Model not found at {checkpoint_path}"

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    print("Top keys in checkpoint:")
    for k in list(checkpoint.keys())[:5]:
        print(k)

    # 尝试加载 class mappings（如果训练脚本没保存，需重建）
    try:
        class_to_idx, idx_to_class = load_class_mappings(OUTPUT_DIR)
        num_classes = len(idx_to_class)
        print(f"Loaded {num_classes} classes from class_mappings.json")
    except Exception as e:
        print(f"class_mappings.json not found ({e}). Rebuilding from dataset...")
        dataset = load_dataset(DATASET_NAME, SUBSET_NAME)
        all_data = []
        for split in ["train", "validation", "test"]:
            if split in dataset:
                all_data.extend(dataset[split])
        class_set = set((bool(item["label"]), str(item["cwe"])) for item in all_data)
        class_list = sorted(list(class_set))
        class_to_idx = {cls: idx for idx, cls in enumerate(class_list)}
        idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
        num_classes = len(class_list)

        # 保存映射：使用 "True|CWE-123" 格式的字符串 key
        class_to_idx_serializable = {
            f"{k[0]}|{k[1]}": v for k, v in class_to_idx.items()
        }
        with open(os.path.join(OUTPUT_DIR, "class_mappings.json"), "w") as f:
            json.dump({"class_to_idx": class_to_idx_serializable}, f)

    num_classes = len(class_to_idx)

    # ===== 加载完整模型 =====
    model = VulnerabilityModel(MODEL_NAME, num_classes, ROBERTA_LAYERS_TO_CONCAT).to(
        DEVICE
    )

    # ===== 加载完整 checkpoint =====
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint, strict=True)  # ✅ Now keys match!

    model.eval()

    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

    # 加载训练集（用于计算 prototypes 和 thresholds）
    print("Loading training set to compute prototypes...")
    train_data = load_dataset(DATASET_NAME, SUBSET_NAME)["val"]
    train_dataset = EvalDataset(train_data, tokenizer, MAX_LENGTH)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=False,collate_fn=custom_collate_fn
    )

    # 提取训练集嵌入
    train_embeddings = []
    train_labels = []
    with torch.no_grad():
        for batch in tqdm(train_loader, desc="Encoding train set"):
            embs = model(
                batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE)
            )
            train_embeddings.append(embs.cpu())
            label_indices = [class_to_idx[k] for k in batch["class_key"]]
            train_labels.extend(label_indices)
    train_embeddings = torch.cat(train_embeddings, dim=0)
    train_labels = torch.tensor(train_labels)

    # 计算 prototypes
    prototypes = torch.zeros(num_classes, train_embeddings.size(1))
    for c in range(num_classes):
        mask = train_labels == c
        if mask.any():
            prototypes[c] = train_embeddings[mask].mean(dim=0)
    prototypes = F.normalize(prototypes, p=2, dim=1).to(DEVICE)

    # 计算 thresholds
    print("Computing thresholds per class...")
    thresholds = compute_thresholds(
        train_embeddings, train_labels, prototypes, idx_to_class
    )

    # 加载评估集
    print(f"Loading {EVAL_SPLIT} set for evaluation...")
    if EVAL_SPLIT == "val":
        eval_data = (
            load_dataset(DATASET_NAME, SUBSET_NAME).get("val")
            or load_dataset(DATASET_NAME, SUBSET_NAME)["val"]
        )
    else:
        eval_data = load_dataset(DATASET_NAME, SUBSET_NAME)[EVAL_SPLIT]
    eval_dataset = EvalDataset(eval_data, tokenizer, MAX_LENGTH)
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=BATCH_SIZE, shuffle=False,collate_fn=custom_collate_fn
    )

    # 开始评估
    all_pred_class_indices = []
    all_true_labels = []
    all_true_class_keys = []

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc=f"Evaluating {EVAL_SPLIT}"):
            embs = model(
                batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE)
            )
            sims = torch.mm(embs, prototypes.T)  # (B, C)
            max_sim, pred_class = sims.max(dim=1)

            for i in range(embs.size(0)):
                tau = thresholds[pred_class[i].item()]
                if max_sim[i] >= tau:
                    all_pred_class_indices.append(pred_class[i].item())
                else:
                    all_pred_class_indices.append(-1)  # unknown

            all_true_labels.extend(batch["label"])
            all_true_class_keys.extend(batch["class_key"])

    # 转为二分类预测
    y_true_binary = np.array(all_true_labels)
    y_pred_binary = []
    for pred_idx in all_pred_class_indices:
        if pred_idx == -1:
            y_pred_binary.append(False)
        else:
            pred_label = idx_to_class[pred_idx][0]
            y_pred_binary.append(pred_label)
    y_pred_binary = np.array(y_pred_binary)

        # ===== 新增：CWE 多分类评估（仅针对正样本）=====
    y_true_cwe = []
    y_pred_cwe = []

    for pred_idx, true_key in zip(all_pred_class_indices, all_true_class_keys):
        true_label, true_cwe = true_key
        if true_label:  # 只考虑真实为漏洞的样本
            if pred_idx == -1:
                # 预测为 unknown → 视为预测错误（可选：也可跳过，但通常计入 FN）
                y_true_cwe.append(true_cwe)
                y_pred_cwe.append("PREDICTED_AS_UNKNOWN")  # 虚拟类别
            else:
                pred_label, pred_cwe = idx_to_class[pred_idx]
                if pred_label:
                    # 预测为漏洞 → 记录 CWE
                    y_true_cwe.append(true_cwe)
                    y_pred_cwe.append(pred_cwe)
                else:
                    # 真实是漏洞，但预测为非漏洞 → CWE 错误
                    y_true_cwe.append(true_cwe)
                    y_pred_cwe.append("PREDICTED_AS_NON_VULN")

    # 获取所有唯一 CWE 类别（用于排序和报告）
    unique_cwes = sorted(set(y_true_cwe))

    # 计算 per-class metrics
    from sklearn.metrics import classification_report

    # 生成分类报告（包含 per-class 和 macro/micro）
    report = classification_report(
        y_true_cwe, y_pred_cwe, labels=unique_cwes, zero_division=0, output_dict=True
    )

    # 提取 macro 和 micro
    macro_precision = report["macro avg"]["precision"]
    macro_recall = report["macro avg"]["recall"]
    macro_f1 = report["macro avg"]["f1-score"]
    micro_precision = report["weighted avg"]["precision"]  # 注意：micro = weighted when balanced
    micro_recall = report["weighted avg"]["recall"]
    micro_f1 = report["weighted avg"]["f1-score"]

    # 但严格来说，micro 应该用 total TP / (TP+FP) 等，sklearn 的 "micro" 需要显式指定
    # 更准确的做法：
    from sklearn.metrics import precision_recall_fscore_support
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        y_true_cwe, y_pred_cwe, average="micro", zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true_cwe, y_pred_cwe, average="macro", zero_division=0
    )

    # Per-class metrics
    per_class_metrics = {}
    for cwe in unique_cwes:
        if cwe in report:
            per_class_metrics[cwe] = {
                "precision": report[cwe]["precision"],
                "recall": report[cwe]["recall"],
                "f1-score": report[cwe]["f1-score"],
                "support": report[cwe]["support"],
            }

    # 保存 CWE 评估结果
    cwe_metrics = {
        "per_class": per_class_metrics,
        "macro": {
            "precision": float(macro_p),
            "recall": float(macro_r),
            "f1": float(macro_f1),
        },
        "micro": {
            "precision": float(micro_p),
            "recall": float(micro_r),
            "f1": float(micro_f1),
        },
    }

    cwe_output_file = os.path.join(OUTPUT_DIR, f"cwe_metrics_{EVAL_SPLIT}.json")
    with open(cwe_output_file, "w") as f:
        json.dump(cwe_metrics, f, indent=4)

    print(f"\nCWE Multi-class Metrics ({EVAL_SPLIT}):")
    print(f"Macro F1: {macro_f1:.4f}, Micro F1: {micro_f1:.4f}")
    print(f"Results saved to: {cwe_output_file}")

    # 计算指标
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_binary, y_pred_binary, average="binary", zero_division=0
    )
    acc = accuracy_score(y_true_binary, y_pred_binary)
    tn, fp, fn, tp = confusion_matrix(
        y_true_binary, y_pred_binary, labels=[False, True]
    ).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    mcc = matthews_corrcoef(y_true_binary, y_pred_binary)

    metrics = {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "mcc": float(mcc),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }

    # 保存结果
    eval_output_file = os.path.join(OUTPUT_DIR, f"metrics_{EVAL_SPLIT}_reval.json")
    with open(eval_output_file, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"\n{EVAL_SPLIT} Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    print(f"\nResults saved to: {eval_output_file}")


if __name__ == "__main__":
    main()
