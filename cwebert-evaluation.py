import json
import os
import warnings
from collections import Counter
from typing import Dict, List, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
    classification_report,
)
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
from umap import UMAP

warnings.filterwarnings("ignore")

# ========================
# 配置：请根据你的训练设置修改以下路径和参数
# ========================

MODEL_DIR = "output_primevul-paired_20251212-14-07-12"  # 👈 修改这里！
EVALUATION_OUTPUT_DIR = os.path.join(MODEL_DIR, "evaluation")

DATASET_NAME = "codemetic/curve"
SUBSET_NAME = "primevul-paired"
MODEL_NAME = "codemetic/CweBERT-mlm"
MAX_LENGTH = 512
BATCH_SIZE = 64
ROBERTA_LAYERS_TO_CONCAT = (6, 7, 8, 9)

UMAP_MAX_POINTS = 10000
EVAL_SPLIT = "test"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(EVALUATION_OUTPUT_DIR, exist_ok=True)

# ========================
# 模型定义
# ========================


class KappaFaceHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        # Not used in eval; placeholder only
        return x


class RoBERTaEncoder(nn.Module):
    def __init__(
        self, model_name: str, layers_to_concat: Tuple[int, ...] = (6, 7, 8, 9)
    ):
        super().__init__()
        config = RobertaConfig.from_pretrained(model_name, output_hidden_states=True)
        self.roberta = RobertaModel.from_pretrained(model_name, config=config)
        self.layers_to_concat = layers_to_concat
        self.hidden_size = config.hidden_size
        self.concat_dim = len(layers_to_concat) * self.hidden_size

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states
        selected_layers = [
            hidden_states[-layer_idx] for layer_idx in self.layers_to_concat
        ]
        concatenated = torch.cat(selected_layers, dim=-1)
        return concatenated[:, 0, :]  # [CLS] token


class VulnerabilityModel(nn.Module):
    def __init__(
        self, model_name: str, num_classes: int, layers_to_concat: Tuple[int, ...]
    ):
        super().__init__()
        self.encoder = RoBERTaEncoder(model_name, layers_to_concat)
        self.kappaface_head = KappaFaceHead(self.encoder.concat_dim, num_classes)

    def forward(self, input_ids, attention_mask):
        features = self.encoder(input_ids, attention_mask)
        return F.normalize(features, p=2, dim=-1)


# ========================
# 数据集类
# ========================


class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_length: int):
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


def custom_collate_fn(batch):
    """Prevent stacking of non-tensor fields."""
    result = {}
    for key in batch[0]:
        if key in ["class_key", "cwe", "label"]:
            result[key] = [d[key] for d in batch]
        else:
            result[key] = torch.stack([d[key] for d in batch])
    return result


# ========================
# 工具函数
# ========================


def build_class_mappings(dataset_name: str, subset_name: str):
    # Rebuild class mappings from full dataset
    print("Rebuilding class mappings from dataset...")
    dataset = load_dataset(dataset_name, subset_name)
    all_items = []
    for split in ["train", "val", "test"]:
        if split in dataset:
            all_items.extend(dataset[split])
    class_set = {(bool(item["label"]), str(item["cwe"])) for item in all_items}
    class_list = sorted(class_set)
    class_to_idx = {cls: idx for idx, cls in enumerate(class_list)}
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}

    return class_to_idx, idx_to_class


def extract_embeddings(
    model: nn.Module,
    dataloader: DataLoader,
    class_to_idx: Dict[Tuple[bool, str], int],
    device: torch.device,
) -> Tuple[torch.Tensor, List[int]]:
    embeddings = []
    labels = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            embs = model(
                batch["input_ids"].to(device), batch["attention_mask"].to(device)
            )
            embeddings.append(embs.cpu())
            label_indices = [class_to_idx[k] for k in batch["class_key"]]
            labels.extend(label_indices)
    return torch.cat(embeddings, dim=0), labels


def compute_prototypes(
    embeddings: torch.Tensor, labels: List[int], num_classes: int
) -> torch.Tensor:
    labels = torch.tensor(labels)
    prototypes = torch.zeros(num_classes, embeddings.size(1))
    for c in range(num_classes):
        mask = labels == c
        if mask.any():
            prototypes[c] = embeddings[mask].mean(dim=0)
    return F.normalize(prototypes, p=2, dim=1)


def compute_thresholds(
    embeddings: torch.Tensor,
    labels: List[int],
    prototypes: torch.Tensor,
    device: torch.device,
) -> Dict[int, float]:
    labels = torch.tensor(labels).cpu().numpy()
    thresholds = {}
    for c in range(prototypes.size(0)):
        mask = labels == c
        if not mask.any():
            thresholds[c] = -1.0
            continue

        class_embs = torch.tensor(embeddings[mask].cpu().numpy(), device=device)
        sims = (
            torch.mm(class_embs, prototypes[c].unsqueeze(0).T).squeeze(1).cpu().numpy()
        )

        best_f1, best_tau = 0.0, 0.5
        for tau in np.linspace(0.0, 1.0, 100):
            preds = (sims >= tau).astype(int)
            if preds.sum() == 0:
                continue
            f1 = precision_recall_fscore_support(
                np.ones_like(preds), preds, average="binary", zero_division=0
            )[2]
            if f1 > best_f1:
                best_f1, best_tau = f1, tau
        thresholds[c] = best_tau
    return thresholds


# ========================
# 可视化
# ========================


def draw_plot_umap(embeddings: np.ndarray, class_keys: List[Tuple[bool, str]]):
    total_samples = len(class_keys)
    stratify_labels = np.array([f"{ck[1]}_{ck[0]}" for ck in class_keys])

    if total_samples > UMAP_MAX_POINTS:
        split = StratifiedShuffleSplit(
            n_splits=1, train_size=UMAP_MAX_POINTS, random_state=42
        )
        sampled_idx, _ = next(split.split(np.zeros(total_samples), stratify_labels))
    else:
        sampled_idx = np.arange(total_samples)

    sampled_embs = embeddings[sampled_idx]
    sampled_keys = [class_keys[i] for i in sampled_idx]

    reducer = UMAP(n_components=2, random_state=42)
    umap_embs = reducer.fit_transform(sampled_embs)

    labels = np.array([ck[0] for ck in sampled_keys])
    cwes = np.array([ck[1] for ck in sampled_keys])
    vuln_mask = labels.astype(bool)

    cwe_unique = sorted(set(cwes[vuln_mask])) if vuln_mask.any() else []
    cwe_to_color = {
        cwe: color
        for cwe, color in zip(cwe_unique, sns.color_palette("husl", len(cwe_unique)))
    }

    fig_width = max(10, 8 + 0.4 * (len(cwe_unique) + 1))
    plt.figure(figsize=(fig_width, 8))
    x, y = umap_embs[:, 0], umap_embs[:, 1]

    if not vuln_mask.all():
        non_vuln = ~vuln_mask
        plt.scatter(
            x[non_vuln],
            y[non_vuln],
            c="lightgray",
            marker="x",
            s=20,
            label="_nolegend_",
        )

    for cwe in cwe_unique:
        mask = (cwes == cwe) & vuln_mask
        if mask.any():
            plt.scatter(
                x[mask], y[mask], c=[cwe_to_color[cwe]], marker="o", s=20, label=cwe
            )

    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title(f"UMAP Projection - Evaluation")
    plt.tight_layout()
    plt.savefig(
        os.path.join(EVALUATION_OUTPUT_DIR, f"umap_evaluation.svg"),
        bbox_inches="tight",
    )
    plt.close()


def draw_prototype_heatmap(
    prototypes: torch.Tensor, idx_to_class: Dict[int, Tuple[bool, str]]
):
    sim_matrix = torch.mm(prototypes, prototypes.t()).cpu().numpy()
    class_labels = [
        f"{ck[1]} ({'V' if ck[0] else 'N'})"
        for ck in [idx_to_class[i] for i in range(len(idx_to_class))]
    ]

    n = len(class_labels)
    size = max(6, n * 0.5)
    plt.figure(figsize=(size, size))
    sns.heatmap(
        sim_matrix,
        xticklabels=class_labels,
        yticklabels=class_labels,
        cmap="viridis",
        square=True,
        cbar_kws={"shrink": 0.8},
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.title(f"Prototype Similarity Heatmap - Evaluation")
    plt.tight_layout()
    plt.savefig(
        os.path.join(EVALUATION_OUTPUT_DIR, f"prototype_heatmap_evaluation.svg"),
        bbox_inches="tight",
    )
    plt.close()


# ========================
# 主流程
# ========================
def main():
    print(f"Loading model from: {MODEL_DIR}")
    model_path = os.path.join(MODEL_DIR, "best_model.pth")
    assert os.path.exists(model_path), f"Model not found at {model_path}"

    # Load class mappings
    class_to_idx, idx_to_class = build_class_mappings(DATASET_NAME, SUBSET_NAME)
    num_classes = len(class_to_idx)

    # Load model
    model = VulnerabilityModel(MODEL_NAME, num_classes, ROBERTA_LAYERS_TO_CONCAT).to(
        DEVICE
    )
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint, strict=True)
    model.eval()

    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

    # Load validation set for prototypes & thresholds
    print("Loading validation set to compute prototypes...")
    val_data = load_dataset(DATASET_NAME, SUBSET_NAME)["val"]
    val_dataset = EvalDataset(val_data, tokenizer, MAX_LENGTH)
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn
    )

    val_embeddings, val_labels_list = extract_embeddings(
        model, val_loader, class_to_idx, DEVICE
    )
    prototypes = compute_prototypes(val_embeddings, val_labels_list, num_classes).to(
        DEVICE
    )
    thresholds = compute_thresholds(val_embeddings, val_labels_list, prototypes, DEVICE)

    # Load evaluation set
    print(f"Loading {EVAL_SPLIT} set for evaluation...")
    test_data = (
        load_dataset(DATASET_NAME, SUBSET_NAME).get(EVAL_SPLIT)
        or load_dataset(DATASET_NAME, SUBSET_NAME)["val"]
    )
    test_dataset = EvalDataset(test_data, tokenizer, MAX_LENGTH)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn
    )

    # Evaluate
    all_pred_class_indices = []
    all_true_labels = []
    all_true_class_keys = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Evaluating {EVAL_SPLIT}"):
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

    # Binary prediction
    y_true_binary = np.array(all_true_labels)
    y_pred_binary = np.array(
        [idx_to_class[idx][0] if idx != -1 else False for idx in all_pred_class_indices]
    )

    # CWE multi-class evaluation (only on true positives)
    y_true_cwe, y_pred_cwe = [], []
    for pred_idx, (true_label, true_cwe) in zip(
        all_pred_class_indices, all_true_class_keys
    ):
        if true_label:
            if pred_idx == -1:
                y_true_cwe.append(true_cwe)
                y_pred_cwe.append("PREDICTED_AS_UNKNOWN")
            else:
                pred_label, pred_cwe = idx_to_class[pred_idx]
                if pred_label:
                    y_true_cwe.append(true_cwe)
                    y_pred_cwe.append(pred_cwe)
                else:
                    y_true_cwe.append(true_cwe)
                    y_pred_cwe.append("PREDICTED_AS_NON_VULN")

    unique_cwes = sorted(set(y_true_cwe))
    if unique_cwes:
        micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
            y_true_cwe, y_pred_cwe, average="micro", zero_division=0
        )
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            y_true_cwe, y_pred_cwe, average="macro", zero_division=0
        )
        per_class_report = classification_report(
            y_true_cwe,
            y_pred_cwe,
            labels=unique_cwes,
            zero_division=0,
            output_dict=True,
        )
        per_class_metrics = {
            cwe: {
                "precision": per_class_report[cwe]["precision"],
                "recall": per_class_report[cwe]["recall"],
                "f1-score": per_class_report[cwe]["f1-score"],
                "support": per_class_report[cwe]["support"],
            }
            for cwe in unique_cwes
            if cwe in per_class_report
        }
    else:
        micro_p = micro_r = micro_f1 = macro_p = macro_r = macro_f1 = 0.0
        per_class_metrics = {}

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

    cwe_output_file = os.path.join(
        EVALUATION_OUTPUT_DIR, f"cwe_metrics_{EVAL_SPLIT}.json"
    )
    with open(cwe_output_file, "w") as f:
        json.dump(cwe_metrics, f, indent=4)

    # Save embeddings for UMAP (use full eval set for consistency)
    test_embeddings, _ = extract_embeddings(model, test_loader, class_to_idx, DEVICE)
    draw_plot_umap(test_embeddings.numpy(), all_true_class_keys)
    draw_prototype_heatmap(prototypes, idx_to_class)

    print(f"\nCWE Multi-class Metrics ({EVAL_SPLIT}):")
    print(f"Macro F1: {macro_f1:.4f}, Micro F1: {micro_f1:.4f}")
    print(f"Results saved to: {cwe_output_file}")

    # Binary metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_binary, y_pred_binary, average="binary", zero_division=0
    )
    acc = accuracy_score(y_true_binary, y_pred_binary)
    tn, fp, fn, tp = confusion_matrix(
        y_true_binary, y_pred_binary, labels=[False, True]
    ).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    mcc = matthews_corrcoef(y_true_binary, y_pred_binary)

    binary_metrics = {
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

    eval_output_file = os.path.join(
        EVALUATION_OUTPUT_DIR, f"metrics_{EVAL_SPLIT}_reval.json"
    )
    with open(eval_output_file, "w") as f:
        json.dump(binary_metrics, f, indent=4)

    print(f"\n{EVAL_SPLIT} Binary Metrics:")
    for k, v in binary_metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    print(f"\nResults saved to: {eval_output_file}")


if __name__ == "__main__":
    main()
