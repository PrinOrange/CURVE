import json
import math
import os
import warnings
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    normalized_mutual_info_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
from umap import UMAP
import random
import time
import seaborn as sns

warnings.filterwarnings("ignore")

# ========================
# Configuration Constants
# ========================

# bigvul,diversevul,vuldeepecker,primevul,primevul-paired,megavul,reposvul
SUBSET_NAME = "primevul-paired"
DATASET_NAME = "codemetic/curve"
MODEL_NAME = "codemetic/CweBERT-mlm"

# Layers: 0 1 2 3 4 5 6 7 8 9 10 11
# The outer layer has more code structure features while the inner layer has more semantic features.
ROBERTA_LAYERS_TO_CONCAT = (6, 7, 8, 9)

MAX_LENGTH = 512
BATCH_SIZE = 64
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
MAX_EPOCHS = 1800
EARLY_STOPPING_PATIENCE = int(MAX_EPOCHS * 0.1)
RANDOM_SEED = 42

# KappaFace Hyperparameters
GAMMA = 0.9  # balance between difficulty and imbalance
TEMPERATURE_T = 0.4  # temperature for difficulty weight
M0 = 0.8  # base margin
S = 30.0  # scale factor
LAMBDA_PROTO = 0.8  # weight for prototype loss

# Memory Buffer vs Momentum Encoder
USE_MOMENTUM_ENCODER = (
    False  # Set to True to use momentum encoder, False for memory buffer
)
ALPHA = 0.3  # memory buffer momentum
MOMENTUM = 0.999  # momentum encoder coefficient

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.set_default_device(DEVICE)

# Image settings
UMAP_MAX_POINTS = 10000

# Output directory
OUTPUT_DIR = f"output_{SUBSET_NAME}_{time.strftime('%Y%m%d-%H%M%S')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ========================
# Utility Functions
# ========================


def set_seed():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)
        torch.cuda.manual_seed_all(RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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


def l2_norm(x: torch.Tensor) -> torch.Tensor:
    """L2 normalize tensor along last dimension"""
    return F.normalize(x, p=2, dim=-1)


def cosine_similarity_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute cosine similarity between x and y (both normalized)"""
    return torch.sum(x * y, dim=-1)


def estimate_vmf_concentration(embeddings: torch.Tensor) -> float:
    """
    Estimate vMF concentration parameter κ from normalized embeddings.
    embeddings: (N, D)
    """
    if embeddings.size(0) == 0:
        return 0.0
    mean_vec = torch.mean(embeddings, dim=0)
    r = torch.norm(mean_vec).item()
    d = embeddings.size(1)
    if r >= 1.0:
        r = 0.999999
    if r <= 0:
        return 0.0
    kappa = r * (d - r * r) / (1 - r * r)
    return max(kappa, 1e-6)


def compute_ch_score(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Calinski-Harabasz Index"""
    if len(np.unique(labels)) <= 1:
        return 0.0
    from sklearn.metrics import calinski_harabasz_score

    return calinski_harabasz_score(embeddings, labels)


# ========================
# Dataset Class
# ========================
class VulnerabilityDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=MAX_LENGTH):
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
# Model Definition
# ========================
class RoBERTaEncoder(nn.Module):
    def __init__(self, model_name, layers_to_concat=ROBERTA_LAYERS_TO_CONCAT):
        super().__init__()
        self.config = RobertaConfig.from_pretrained(
            model_name, output_hidden_states=True
        )
        self.roberta = RobertaModel.from_pretrained(model_name, config=self.config)
        self.layers_to_concat = layers_to_concat  # These will be counted from the end
        self.hidden_size = self.config.hidden_size
        self.concat_dim = len(layers_to_concat) * self.hidden_size

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        # hidden_states is a tuple of (layer0, layer1, ..., layer12)
        hidden_states = outputs.hidden_states  # tuple of 13 tensors
        # We want layers [-9, -8, -7, -6] → indexes [4,5,6,7] (since len=13)
        selected_layers = []
        for layer_idx in self.layers_to_concat:
            actual_idx = -(layer_idx)  # e.g., -6 → index 7 in 0-based 13-length list
            selected_layers.append(hidden_states[actual_idx])
        # Concatenate along last dimension
        concatenated = torch.cat(selected_layers, dim=-1)  # (B, L, 768*4)
        # Use [CLS] token
        cls_embedding = concatenated[:, 0, :]  # (B, 768*4)
        return cls_embedding


class KappaFaceHead(nn.Module):
    def __init__(self, in_features, num_classes, s=S, m0=M0):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.s = s
        self.m0 = m0
        self.weight = nn.Parameter(torch.Tensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, labels, margins):
        """
        x: (B, D) normalized features
        labels: (B,) class indices
        margins: (num_classes,) dynamic margins per class
        """
        # Normalize weight
        weight_norm = l2_norm(self.weight)
        # Cosine similarity matrix: (B, C)
        cos_theta = torch.mm(x, weight_norm.t())
        # Clamp for numerical stability
        cos_theta = cos_theta.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        # Add margins to true class
        one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        margins_expanded = margins.unsqueeze(0).expand_as(cos_theta)
        cos_theta_m = cos_theta - one_hot * margins_expanded

        # Scale
        logits = self.s * cos_theta_m
        return logits


class VulnerabilityModel(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.encoder = RoBERTaEncoder(model_name)
        self.feature_dim = self.encoder.concat_dim
        self.kappaface_head = KappaFaceHead(self.feature_dim, num_classes)
        self.prototypes = None  # Will be set during training

    def forward(self, input_ids, attention_mask, labels=None, margins=None):
        features = self.encoder(input_ids, attention_mask)
        features_norm = l2_norm(features)
        if labels is not None and margins is not None:
            logits = self.kappaface_head(features_norm, labels, margins)
            return features_norm, logits
        else:
            return features_norm


# ========================
# Loss Functions
# ========================
def kappaface_loss(logits, labels):
    return F.cross_entropy(logits, labels)


def prototype_loss(embeddings, labels, prototypes, num_classes):
    """
    embeddings: (N, D) normalized
    labels: (N,) class indices
    prototypes: (C, D) normalized prototypes
    """
    loss = 0.0
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() == 0:
            continue
        class_embs = embeddings[mask]
        proto = prototypes[c].unsqueeze(0)  # (1, D)
        # Cosine similarity
        cos_sim = torch.sum(class_embs * proto, dim=1)  # (N_c,)
        loss += torch.sum(1.0 - cos_sim)
    return 2.0 * loss / embeddings.size(0)


# ========================
# Trainer Class
# ========================
class Trainer:
    def __init__(
        self, train_dataset, val_dataset, test_dataset, class_to_idx, idx_to_class
    ):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.class_to_idx = class_to_idx
        self.idx_to_class = idx_to_class
        self.num_classes = len(class_to_idx)

        self.model = VulnerabilityModel(MODEL_NAME, self.num_classes).to(DEVICE)
        if USE_MOMENTUM_ENCODER:
            self.momentum_model = VulnerabilityModel(MODEL_NAME, self.num_classes).to(
                DEVICE
            )
            self.momentum_model.load_state_dict(self.model.state_dict())
            for param in self.momentum_model.parameters():
                param.requires_grad = False
        else:
            self.memory_buffer = None  # Will be initialized with train size

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )

        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.global_step = 0

        # For dynamic margin
        self.class_counts = self._count_classes(train_dataset)
        self.max_count = max(self.class_counts.values())

        # Build sample index to buffer row mapping (identity due to shuffle=False)
        self.sample_to_buffer_idx = {i: i for i in range(len(train_dataset))}
        self.total_train_samples = len(train_dataset)

    def _count_classes(self, dataset):
        counts = Counter()
        for item in dataset.data:
            key = (bool(item["label"]), str(item["cwe"]))
            counts[key] += 1
        return {self.class_to_idx[k]: v for k, v in counts.items()}

    def _build_memory_buffer(self, dataloader):
        """Initialize memory buffer for all training samples"""
        total_samples = len(self.train_dataset)
        feature_dim = self.model.feature_dim
        buffer = torch.zeros(total_samples, feature_dim, device=DEVICE)
        labels = torch.zeros(total_samples, dtype=torch.long, device=DEVICE)
        idx_map = {}
        current_idx = 0
        for batch in tqdm(dataloader, desc="Building memory buffer"):
            with torch.no_grad():
                features = self.model.encoder(
                    batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE)
                )
                features_norm = l2_norm(features)
                batch_size = features_norm.size(0)
                buffer[current_idx : current_idx + batch_size] = features_norm
                label_indices = [self.class_to_idx[k] for k in batch["class_key"]]
                labels[current_idx : current_idx + batch_size] = torch.tensor(
                    label_indices, device=DEVICE
                )
                for i in range(batch_size):
                    idx_map[current_idx + i] = (
                        batch["input_ids"][i],
                        batch["attention_mask"][i],
                    )
                current_idx += batch_size
        return buffer, labels, idx_map

    def _compute_dynamic_margins(self, embeddings_or_buffer, labels):
        """Compute dynamic margins for all classes"""
        margins = torch.zeros(self.num_classes, device=DEVICE)
        kappas = []
        class_r = {}

        for c in range(self.num_classes):
            mask = labels == c
            if mask.sum() == 0:
                kappas.append(1e-6)
                class_r[c] = 0.0
                continue
            class_embs = embeddings_or_buffer[mask]
            kappa = estimate_vmf_concentration(class_embs)
            kappas.append(kappa)
            class_r[c] = torch.norm(torch.mean(class_embs, dim=0)).item()

        # Normalize kappas
        kappas = np.array(kappas)
        if len(kappas) > 1:
            mu_k = np.mean(kappas)
            sigma_k = np.std(kappas) + 1e-8
            kappas_norm = (kappas - mu_k) / sigma_k
        else:
            kappas_norm = np.zeros_like(kappas)

        for c in range(self.num_classes):
            # Difficulty weight
            omega_k = 1.0 / (1.0 + math.exp(kappas_norm[c] / TEMPERATURE_T))
            # Imbalance weight
            n_c = self.class_counts.get(c, 1)
            omega_n = self.max_count / n_c
            # Dynamic margin
            m_c = M0 * (GAMMA * omega_k + (1 - GAMMA) * omega_n)
            margins[c] = m_c

        return margins

    def _compute_prototypes(self, embeddings, labels):
        """Compute prototypes for each class"""
        prototypes = torch.zeros(self.num_classes, embeddings.size(1), device=DEVICE)
        for c in range(self.num_classes):
            mask = labels == c
            if mask.sum() == 0:
                prototypes[c] = torch.zeros(embeddings.size(1), device=DEVICE)
            else:
                prototypes[c] = torch.mean(embeddings[mask], dim=0)
        return l2_norm(prototypes)

    def _evaluate_epoch(self, dataloader, split_name, epoch):
        self.model.eval()
        all_embeddings = []
        all_labels = []
        all_class_keys = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {split_name}"):
                embeddings = self.model(
                    batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE)
                )
                all_embeddings.append(embeddings.cpu())
                label_indices = [self.class_to_idx[k] for k in batch["class_key"]]
                all_labels.extend(label_indices)
                all_class_keys.extend(batch["class_key"])

        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_embeddings_np = all_embeddings.numpy()
        all_labels_np = np.array(all_labels)

        # Compute clustering metrics
        ch_score = compute_ch_score(all_embeddings_np, all_labels_np)

        if len(np.unique(all_labels_np)) > 1:
            kmeans = KMeans(n_clusters=len(np.unique(all_labels_np)), random_state=42)
            cluster_labels = kmeans.fit_predict(all_embeddings_np)
            nmi_score = normalized_mutual_info_score(all_labels_np, cluster_labels)
        else:
            nmi_score = 0.0  # or np.nan, but 0 is safer for logging

        # Plot UMAP
        self._plot_umap(all_embeddings_np, all_class_keys, split_name, epoch)

        all_embeddings_gpu = all_embeddings.to(DEVICE)
        all_labels_tensor = torch.tensor(all_labels, device=DEVICE)
        prototypes = self._compute_prototypes(all_embeddings_gpu, all_labels_tensor)

        # 构造 class labels（字符串形式，用于 heatmap 显示）
        unique_labels = sorted(set(all_labels))
        class_labels = [
            f"{self.idx_to_class[idx][1]}_{self.idx_to_class[idx][0]}"
            for idx in unique_labels
        ]

        # 绘制热力图
        self._plot_prototype_heatmap(prototypes, class_labels, split_name, epoch)

        return ch_score, nmi_score

    def _plot_umap(self, embeddings, class_keys, split_name, epoch=None):
        # ------------------------------------------
        # 1. 分层抽样
        # ------------------------------------------
        total_samples = len(class_keys)
        stratify_labels = np.array([f"{ck[1]}_{ck[0]}" for ck in class_keys])

        if total_samples > UMAP_MAX_POINTS:
            split = StratifiedShuffleSplit(
                n_splits=1, train_size=UMAP_MAX_POINTS, random_state=42
            )
            sampled_idx, _ = next(split.split(np.zeros(total_samples), stratify_labels))
        else:
            sampled_idx = np.arange(total_samples)

        sampled_embeddings = embeddings[sampled_idx]
        sampled_class_keys = [class_keys[i] for i in sampled_idx]

        # ------------------------------------------
        # 2. UMAP 降维
        # ------------------------------------------
        reducer = UMAP(n_components=2, random_state=42)
        umap_embeddings = reducer.fit_transform(sampled_embeddings)

        labels = np.array([ck[0] for ck in sampled_class_keys])  # bool array
        cwes = np.array([ck[1] for ck in sampled_class_keys])

        # ------------------------------------------
        # 3. 颜色映射（仅漏洞样本按 CWE 上色）
        # ------------------------------------------
        vuln_mask = labels.astype(bool)
        vulnerable_cwes = cwes[vuln_mask]
        cwe_unique_vuln = sorted(set(vulnerable_cwes))

        if cwe_unique_vuln:
            palette = sns.color_palette("husl", n_colors=len(cwe_unique_vuln))
            cwe_to_color = {cwe: palette[i] for i, cwe in enumerate(cwe_unique_vuln)}
        else:
            cwe_to_color = {}

        # ------------------------------------------
        # 4. 自适应 figsize
        # ------------------------------------------
        num_legend_items = len(cwe_unique_vuln) + 1  # +1 for "Non-vulnerable"
        fig_width = max(10, 8 + 0.4 * num_legend_items)
        fig_height = 8
        plt.figure(figsize=(fig_width, fig_height))

        x = umap_embeddings[:, 0]
        y = umap_embeddings[:, 1]

        # ------------------------------------------
        # 5. 分层绘制：先画非漏洞（底层），再画漏洞（上层）
        # ------------------------------------------
        non_vuln_mask = ~vuln_mask
        # 非漏洞：灰色 x
        if np.any(non_vuln_mask):
            plt.scatter(
                x[non_vuln_mask],
                y[non_vuln_mask],
                c="lightgray",
                marker="x",
                s=20,
                edgecolors="none",
                label="_nolegend_",
            )

        # 漏洞：按 CWE 上色，圆圈
        for cwe in cwe_unique_vuln:
            cwe_mask = (cwes == cwe) & vuln_mask
            if np.any(cwe_mask):
                plt.scatter(
                    x[cwe_mask],
                    y[cwe_mask],
                    c=[cwe_to_color[cwe]],
                    marker="o",
                    s=20,
                    edgecolors="none",
                    label=cwe,
                )

        # ------------------------------------------
        # 6. 图例 + 保存
        # ------------------------------------------
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        if epoch is None:
            plt.title(f"UMAP Projection - {split_name}")
            output_path = os.path.join(OUTPUT_DIR, f"umap_{split_name}.svg")
        else:
            plt.title(f"UMAP Projection - {split_name} (Epoch {epoch})")
            output_path = os.path.join(
                OUTPUT_DIR, f"umap_{split_name}_epoch_{epoch}.svg"
            )

        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

    def _plot_prototype_heatmap(self, prototypes, class_labels, split_name, epoch=None):
        sim_matrix = torch.mm(prototypes, prototypes.t()).cpu().numpy()
        n_classes = len(class_labels)

        # 自适应 figsize：每个类别占 ~0.5 英寸，至少 6x6
        cell_size = 0.5
        fig_width = max(6, n_classes * cell_size)
        fig_height = max(6, n_classes * cell_size)

        plt.figure(figsize=(fig_width, fig_height))
        sns.heatmap(
            sim_matrix,
            xticklabels=class_labels,
            yticklabels=class_labels,
            cmap="viridis",
            square=True,  # 保证单元格为正方形
            cbar_kws={"shrink": 0.8},  # 色标条适当缩小
        )
        plt.title(f"Prototype Similarity Heatmap - {split_name}")

        # 自动调整标签旋转（可选，提升可读性）
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

        plt.tight_layout()

        if epoch is None:
            output_path = os.path.join(
                OUTPUT_DIR, f"prototype_heatmap_{split_name}.svg"
            )
        else:
            output_path = os.path.join(
                OUTPUT_DIR, f"prototype_heatmap_{split_name}_epoch_{epoch}.svg"
            )

        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

    def _compute_thresholds(self, train_embeddings, train_labels):
        """Compute threshold τ per class to maximize F1"""
        thresholds = {}
        prototypes = self._compute_prototypes(train_embeddings, train_labels)
        train_embeddings_np = train_embeddings.cpu().numpy()
        train_labels_np = train_labels.cpu().numpy()

        for c in range(self.num_classes):
            mask = train_labels_np == c
            if not np.any(mask):
                thresholds[c] = 0.5
                continue
            class_embs = train_embeddings_np[mask]
            sims = (
                cosine_similarity_torch(
                    torch.tensor(class_embs, device=DEVICE), prototypes[c].unsqueeze(0)
                )
                .cpu()
                .numpy()
            )
            # Try thresholds
            best_f1 = 0
            best_tau = 0.5
            for tau in np.linspace(0.0, 1.0, 100):
                preds = (sims >= tau).astype(int)
                if preds.sum() == 0:
                    continue
                f1 = precision_recall_fscore_support(
                    np.ones_like(preds), preds, average="binary", zero_division=0
                )[2]
                if f1 > best_f1:
                    best_f1 = f1
                    best_tau = tau
            thresholds[c] = best_tau
        return thresholds, prototypes

    def _evaluate_final(self, dataloader, thresholds, prototypes, split_name):
        self.model.eval()
        all_embeddings = []
        all_true_labels = []
        all_true_class_keys = []
        all_pred_class_indices = []
        all_labels = []
        all_class_keys = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Final {split_name} evaluation"):
                embeddings = self.model(
                    batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE)
                )
                all_embeddings.append(embeddings)
                all_true_labels.extend(batch["label"])
                label_indices = [self.class_to_idx[k] for k in batch["class_key"]]
                all_labels.extend(label_indices)
                all_true_class_keys.extend(batch["class_key"])

        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_pred_class_indices = []

        # Predict using prototype distances
        for i in range(all_embeddings.size(0)):
            emb = all_embeddings[i : i + 1]  # (1, D)
            sims = torch.mm(emb, prototypes.t()).squeeze(0)  # (C,)
            max_sim, pred_class = torch.max(sims, dim=0)
            tau = thresholds[pred_class.item()]
            if max_sim >= tau:
                all_pred_class_indices.append(pred_class.item())
            else:
                all_pred_class_indices.append(-1)  # unknown

        # Convert to binary predictions
        y_true_binary = np.array(all_true_labels)
        y_pred_binary = []
        for pred_idx, true_key in zip(all_pred_class_indices, all_true_class_keys):
            if pred_idx == -1:
                y_pred_binary.append(False)  # unknown → treat as non-vuln
            else:
                pred_key = self.idx_to_class[pred_idx]
                y_pred_binary.append(pred_key[0])  # label=True/False

        y_pred_binary = np.array(y_pred_binary)

        # Metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_binary, y_pred_binary, average="binary", zero_division=0
        )
        acc = accuracy_score(y_true_binary, y_pred_binary)
        tn, fp, fn, tp = confusion_matrix(
            y_true_binary, y_pred_binary, labels=[False, True]
        ).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
        mcc = matthews_corrcoef(y_true_binary, y_pred_binary)

        all_embeddings_np = all_embeddings.numpy()
        all_labels_np = np.array(all_labels)

        # Plot UMAP
        self._plot_umap(all_embeddings_np, all_class_keys, split_name)

        all_embeddings_gpu = all_embeddings.to(DEVICE)
        all_labels_tensor = torch.tensor(all_labels, device=DEVICE)
        prototypes = self._compute_prototypes(all_embeddings_gpu, all_labels_tensor)

        # 构造 class labels（字符串形式，用于 heatmap 显示）
        unique_labels = sorted(set(all_labels))
        class_labels = [
            f"{self.idx_to_class[idx][1]}_{self.idx_to_class[idx][0]}"
            for idx in unique_labels
        ]

        # 绘制热力图
        self._plot_prototype_heatmap(prototypes, class_labels, split_name)

        metrics = {
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "specificity": float(specificity),
            "fpr": float(fpr),
            "fnr": float(fnr),
            "tpr": float(tpr),
            "tnr": float(tnr),
            "mcc": float(mcc),
        }

        # Save metrics
        with open(os.path.join(OUTPUT_DIR, f"metrics_{split_name}.json"), "w") as f:
            json.dump(metrics, f, indent=4)

        return metrics

    def train(self):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=custom_collate_fn,
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=custom_collate_fn,
        )

        if not USE_MOMENTUM_ENCODER:
            print("Building initial memory buffer...")
            self.memory_buffer, self.memory_labels, _ = self._build_memory_buffer(
                train_loader
            )

        for epoch in range(MAX_EPOCHS):
            print(f"\nEpoch {epoch + 1}/{MAX_EPOCHS}")
            self.model.train()

            # Step 1: Compute dynamic margins
            if USE_MOMENTUM_ENCODER:
                # Use momentum model to get stable features
                momentum_embeddings = []
                momentum_labels = []
                with torch.no_grad():
                    for batch in tqdm(train_loader, desc="Computing momentum features"):
                        embs = self.momentum_model.encoder(
                            batch["input_ids"].to(DEVICE),
                            batch["attention_mask"].to(DEVICE),
                        )
                        embs_norm = l2_norm(embs)
                        momentum_embeddings.append(embs_norm)
                        label_indices = [
                            self.class_to_idx[k] for k in batch["class_key"]
                        ]
                        momentum_labels.extend(label_indices)
                momentum_embeddings = torch.cat(momentum_embeddings, dim=0)
                momentum_labels = torch.tensor(momentum_labels, device=DEVICE)
                margins = self._compute_dynamic_margins(
                    momentum_embeddings, momentum_labels
                )
            else:
                # <<< REMOVED _update_memory_buffer CALL HERE >>>
                # Memory buffer is updated in the training loop per batch
                margins = self._compute_dynamic_margins(
                    self.memory_buffer, self.memory_labels
                )

            total_loss = 0.0
            progress_bar = tqdm(train_loader, desc="Training")
            sample_offset = 0  # Track current position in memory buffer
            for batch in progress_bar:
                self.optimizer.zero_grad()

                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                label_indices = [self.class_to_idx[k] for k in batch["class_key"]]
                labels = torch.tensor(label_indices, device=DEVICE)

                features_norm, logits = self.model(
                    input_ids, attention_mask, labels, margins
                )

                loss_kappa = kappaface_loss(logits, labels)
                prototypes = self._compute_prototypes(features_norm.detach(), labels)
                loss_proto = prototype_loss(
                    features_norm, labels, prototypes, self.num_classes
                )
                loss = (1 - LAMBDA_PROTO) * loss_kappa + LAMBDA_PROTO * loss_proto

                loss.backward()
                self.optimizer.step()

                # ============ UPDATE MEMORY BUFFER PER BATCH ============
                if not USE_MOMENTUM_ENCODER:
                    batch_size = features_norm.size(0)
                    start_idx = sample_offset
                    end_idx = start_idx + batch_size
                    with torch.no_grad():
                        self.memory_buffer[start_idx:end_idx] = (
                            ALPHA * self.memory_buffer[start_idx:end_idx]
                            + (1 - ALPHA) * features_norm
                        )
                    sample_offset += batch_size
                # ========================================================

                if USE_MOMENTUM_ENCODER:
                    # Update momentum encoder
                    with torch.no_grad():
                        for param_q, param_k in zip(
                            self.model.parameters(), self.momentum_model.parameters()
                        ):
                            param_k.data = (
                                MOMENTUM * param_k.data + (1 - MOMENTUM) * param_q.data
                            )

                total_loss += loss.item()
                progress_bar.set_postfix({"loss": loss.item()})

            avg_loss = total_loss / len(train_loader)

            # Evaluate on validation set
            ch_score, nmi_score = self._evaluate_epoch(val_loader, "val", epoch + 1)
            print(
                f"""Val CH: {ch_score:.4f}, NMI: {nmi_score:.4f}, 
Avg Loss: {avg_loss:.4f}, History Best Loss: ${self.best_val_loss:.4f}
Loss Kappa: {loss_kappa.item():.4f}, Loss Proto: {loss_proto.item():.4f}"""
            )

            # Early stopping
            if avg_loss < self.best_val_loss:
                self.best_val_loss = avg_loss
                self.patience_counter = 0
                # Save best model
                torch.save(
                    self.model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pth")
                )
            else:
                self.patience_counter += 1
                print(
                    f"Patience Counter: {self.patience_counter}/{EARLY_STOPPING_PATIENCE}"
                )

            if self.patience_counter >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

        # Final evaluation on test set
        print("Loading best model for final evaluation...")
        self.model.load_state_dict(
            torch.load(os.path.join(OUTPUT_DIR, "best_model.pth"))
        )

        # Compute prototypes and thresholds from training set
        # Step 1: Compute prototypes from TRAIN set (model representation)
        self.model.eval()
        train_loader_full = DataLoader(
            self.train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=custom_collate_fn,
        )
        all_train_embs = []
        all_train_labels = []
        with torch.no_grad():
            for batch in tqdm(train_loader_full, desc="Computing train embeddings for prototypes"):
                embs = self.model(
                    batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE)
                )
                all_train_embs.append(embs)
                label_indices = [self.class_to_idx[k] for k in batch["class_key"]]
                all_train_labels.extend(label_indices)
        all_train_embs = torch.cat(all_train_embs, dim=0)
        all_train_labels = torch.tensor(all_train_labels, device=DEVICE)
        prototypes = self._compute_prototypes(all_train_embs, all_train_labels)

        # Step 2: Compute thresholds from VALIDATION set (hyperparameter tuning)
        val_loader_full = DataLoader(
            self.val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=custom_collate_fn,
        )
        all_val_embs = []
        all_val_labels = []
        with torch.no_grad():
            for batch in tqdm(val_loader_full, desc="Computing val embeddings for thresholds"):
                embs = self.model(
                    batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE)
                )
                all_val_embs.append(embs)
                label_indices = [self.class_to_idx[k] for k in batch["class_key"]]
                all_val_labels.extend(label_indices)
        all_val_embs = torch.cat(all_val_embs, dim=0)
        all_val_labels = torch.tensor(all_val_labels, device=DEVICE)
        thresholds = self._compute_thresholds_from_embeddings(
            all_val_embs, all_val_labels, prototypes
        )

        test_loader = DataLoader(
            self.test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=custom_collate_fn,
        )
        test_metrics = self._evaluate_final(test_loader, thresholds, prototypes, "test")
        print("Test Metrics:", test_metrics)

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=custom_collate_fn,
        )
        val_metrics = self._evaluate_final(val_loader, thresholds, prototypes, "val")
        print("Val Metrics:", val_metrics)


# ========================
# Main Execution
# ========================
def main():

    set_seed()

    print("Start training...")
    print("Trainning Arguments:")
    print(f"Dataset: {DATASET_NAME} - {SUBSET_NAME}")
    print(f"Model: {MODEL_NAME}, Layers to Concat: {ROBERTA_LAYERS_TO_CONCAT}")
    print(f"Batch Size: {BATCH_SIZE}, Random Seed: {RANDOM_SEED}")
    print(f"Learning Rate: {LEARNING_RATE}, Weight Decay: {WEIGHT_DECAY}")
    print(f"Max Epochs: {MAX_EPOCHS}")
    print(f"Device: {DEVICE}")
    print(f"Gamma: {GAMMA}, Temperature T: {TEMPERATURE_T}, M0: {M0}, S: {S}")
    print(f"Alpha: {ALPHA}, Momentum: {MOMENTUM}")
    print(f"Prototype Loss Weight: {LAMBDA_PROTO}")
    print(f"Using Momentum Encoder: {USE_MOMENTUM_ENCODER}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("======================================================")

    # Load dataset
    dataset = load_dataset(DATASET_NAME, SUBSET_NAME)
    train_data = dataset["train"]
    val_data = dataset["validation"] if "validation" in dataset else dataset["val"]
    test_data = dataset["test"]

    # Build class mapping
    all_samples = []
    all_samples.extend(train_data)
    all_samples.extend(val_data)
    all_samples.extend(test_data)

    class_set = set()
    for item in all_samples:
        key = (bool(item["label"]), str(item["cwe"]))
        class_set.add(key)
    class_list = sorted(list(class_set))
    class_to_idx = {cls: idx for idx, cls in enumerate(class_list)}
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    print(f"Total classes: {len(class_list)}")

    # Tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

    # Datasets
    train_dataset = VulnerabilityDataset(train_data, tokenizer)
    val_dataset = VulnerabilityDataset(val_data, tokenizer)
    test_dataset = VulnerabilityDataset(test_data, tokenizer)

    # Trainer
    trainer = Trainer(
        train_dataset, val_dataset, test_dataset, class_to_idx, idx_to_class
    )
    trainer.train()


if __name__ == "__main__":
    main()
