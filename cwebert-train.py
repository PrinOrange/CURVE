#!/usr/bin/env python3
# train_modified.py
import math
import os
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from datasets import DatasetDict, load_dataset
from transformers import RobertaModel, RobertaTokenizer

DATASET = "megavul"
SPLIT = "data"


# -------------------------
# Helper: fix batched encodings
# -------------------------
def fix_batch_encoding(enc):
    out = {}
    for k, v in enc.items():
        if torch.is_tensor(v):
            if v.dim() == 3 and v.size(1) == 1:
                out[k] = v.squeeze(1)
            else:
                out[k] = v
        else:
            out[k] = v
    return out


# -------------------------
# Custom collate_fn for DataLoader
# -------------------------
def collate_fn(batch):
    encs = {
        k: torch.stack([item[0][k].squeeze(0) for item in batch]) for k in batch[0][0]
    }
    labels = torch.tensor([item[1] for item in batch])
    cwe_ids = torch.tensor([item[2] for item in batch])
    return encs, labels, cwe_ids


# =========================
# Dataset
# =========================
class CweDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, cwe2id, is_train=True):
        self.tokenizer = tokenizer
        self.samples = []
        self.cwe2id = cwe2id
        self.is_train = is_train

        for item in hf_dataset:
            code = item["source"]
            label = int(item["label"])

            if is_train:
                if label == 1:
                    cwe = item["cwe"]
                    if cwe not in cwe2id:
                        continue
                    cwe_id = cwe2id[cwe]
                else:
                    cwe_id = -1  # negative sample
                self.samples.append((code, label, cwe_id))
            else:
                cwe = item["cwe"] if label == 1 and item.get("cwe", None) else None
                cwe_id = cwe2id[cwe] if (cwe in cwe2id) else -1
                self.samples.append((code, label, cwe_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        code, label, cwe_id = self.samples[idx]
        enc = self.tokenizer(
            code,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return enc, label, cwe_id


# =========================
# ElasticFace (Adaptive ArcFace)
# =========================
class ArcFace(nn.Module):
    def __init__(self, in_features, out_features, adaptive_margins, scale=30.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.scale = scale
        # adaptive_margins expected as tensor [num_classes] (float)
        # store margins as buffer so they move with model.to(device)
        self.register_buffer("margins_buf", adaptive_margins.float())

    def forward(self, x, labels):
        # x: (B, H), labels: (B,)
        x = F.normalize(x, p=2, dim=-1)
        W = F.normalize(self.weight, p=2, dim=-1)

        logits = torch.matmul(x, W.t())  # [B, C]

        # clamp for numerical safety
        logits = logits.clamp(-1 + 1e-7, 1 - 1e-7)

        if labels is None:
            return logits * self.scale

        # Compute theta
        theta = torch.acos(logits)  # [B, C]
        # gather target theta
        target_theta = theta.gather(1, labels.view(-1, 1)).squeeze(1)  # [B]

        # class margins for each example
        m = self.margins_buf[labels]  # [B]

        # new target logit = cos(theta + m)
        new_target = torch.cos(target_theta + m)

        # scatter replacement (make a copy)
        logits_scattered = logits.clone()
        logits_scattered.scatter_(1, labels.view(-1, 1), new_target.unsqueeze(1))

        return logits_scattered * self.scale


# =========================
# CweBERT + ArcFace
# =========================
class CweBertArcFace(nn.Module):
    def __init__(self, model_name, num_cwes, adaptive_margins):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained(model_name)
        H = self.encoder.config.hidden_size
        self.num_cwes = num_cwes
        all_margins = torch.cat(
            [adaptive_margins, torch.tensor([0.0], device=adaptive_margins.device)]
        )
        self.head = ArcFace(H, num_cwes + 1, all_margins, scale=30.0)

    def forward(self, encodings, cwe_ids):
        out = self.encoder(**encodings)
        CLS = out.last_hidden_state[:, 0, :]
        targets = cwe_ids
        targets[targets == -1] = self.num_cwes
        logits = self.head(CLS, targets)
        return logits, CLS

    def encode(self, encodings):
        out = self.encoder(**encodings)
        CLS = out.last_hidden_state[:, 0, :]
        return F.normalize(CLS, p=2, dim=-1)


# =========================
# Adaptive margins
# =========================
def build_adaptive_margins(cwe_counts):
    margins = []
    for cwe, n in cwe_counts.items():
        m = 0.35 + 0.5 / math.sqrt(n + 1)
        margins.append(m)
    return torch.tensor(margins, dtype=torch.float32)


# =========================
# Open-set prediction
# =========================
def open_set_predict(model, encodings, cwe_centers, tau, device):
    model.eval()
    with torch.no_grad():
        encodings = fix_batch_encoding(encodings)
        encodings = {k: v.to(device) for k, v in encodings.items()}
        vec = model.encode(encodings)
        # K E Y   C H A N G E : Exclude the negative class center
        centers = cwe_centers[:-1].to(vec.device)
        sims = torch.matmul(vec, centers.t())
        max_sim, argmax = sims.max(dim=1)
        # preds: 1 if max_sim >= tau (known vulnerable), 0 otherwise (unknown/safe)
        preds = (max_sim >= tau).long().cpu().numpy()
        return preds, argmax.cpu().numpy()


# =========================
# Cache CLS
# =========================
def cache_cls_in_memory(model, loader, device):
    print("===== Caching CLS to memory =====")
    model.eval()
    all_cls, all_labels = [], []
    with torch.no_grad():
        for enc, label, cwe_id in tqdm(loader, desc="Caching CLS"):
            enc = fix_batch_encoding(enc)
            enc = {k: v.to(device) for k, v in enc.items()}
            cls = model.encode(enc)
            all_cls.append(cls.cpu())
            all_labels.append(
                cwe_id.cpu()
                if torch.is_tensor(cwe_id)
                else torch.tensor(cwe_id, dtype=torch.long)
            )
    all_cls = torch.cat(all_cls, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    print(f"Cached {len(all_cls)} CLS vectors in memory.")
    return all_cls, all_labels


def compute_centers_from_cache(cls_cached, labels_cached, num_cwes, device=None):
    if device is None:
        device = cls_cached.device
    hidden = cls_cached.size(1)
    centers = torch.zeros(num_cwes + 1, hidden, device=device)
    labels_cached = labels_cached.to(device)
    cls_cached = cls_cached.to(device)
    for cid in range(num_cwes):
        mask = labels_cached == cid
        if mask.sum() == 0:
            continue
        vec = cls_cached[mask]
        centers[cid] = F.normalize(vec.mean(dim=0), p=2, dim=-1)
    return centers


# =========================
# Center separation loss
# =========================
def center_separation_loss(centers, margin=0.0):
    sim = torch.matmul(centers, centers.t())
    C = sim.shape[0]
    mask = ~torch.eye(C, dtype=torch.bool, device=sim.device)
    sims_offdiag = sim[mask]
    slack = torch.clamp(sims_offdiag - margin, min=0.0)
    loss = (slack**2).mean()
    return loss


def compute_val_loss(model, val_loader, device):
    """Compute validation loss using only vulnerable samples (label==1)"""
    model.eval()
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for enc, label, cwe_id in val_loader:
            # normalize label to numpy
            if torch.is_tensor(label):
                lab_arr = label.cpu().numpy()
            else:
                lab_arr = np.array(label)

            # Only use vulnerable samples
            mask = lab_arr == 1
            if not mask.any():
                continue

            enc = fix_batch_encoding(enc)
            # filter tensors along batch dimension
            enc_vuln = {k: v[mask] for k, v in enc.items()}
            enc_vuln = {k: v.to(device) for k, v in enc_vuln.items()}

            # 关键修改: 确保 cwe_id 是一个 tensor
            if torch.is_tensor(cwe_id):
                cwe_ids_tensor = cwe_id.long().to(device)
            else:
                # 将列表转换为 tensor 并发送到设备
                cwe_ids_tensor = torch.tensor(cwe_id, dtype=torch.long, device=device)

            # 筛选出易受攻击样本的 CWE ID
            cwe_ids_vuln = cwe_ids_tensor[torch.from_numpy(mask).to(device)]

            if len(cwe_ids_vuln) == 0:
                continue

            # 使用张量版本调用模型
            logits, _ = model(enc_vuln, cwe_ids_vuln)

            # 在 CweBertArcFace.forward 中，targets[targets == -1] = self.num_cwes 已经处理了 -1
            # 所以这里直接使用 cross_entropy
            loss = F.cross_entropy(logits, cwe_ids_vuln)

            total_loss += loss.item()
            count += 1

    return total_loss / max(1, count)


def compute_val_cosine_distribution(model, cwe_centers, val_loader, device):
    """Return list of cosine similarities (max_sim) for positive & negative samples"""
    model.eval()
    pos_sims = []
    neg_sims = []

    # K E Y   C H A N G E : Slice to exclude the last (negative class) center
    # The cwe_centers passed in here (ema_centers) has size (num_cwes + 1, H)
    centers_device = cwe_centers[:-1].to(
        device
    )  # <--- Only take the first num_cwes centers

    with torch.no_grad():
        for enc, label, cwe_id in val_loader:
            enc = fix_batch_encoding(enc)
            enc = {k: v.to(device) for k, v in enc.items()}
            vec = model.encode(enc)  # (B, H)

            sims = torch.matmul(vec, centers_device.t())  # (B, num_cwes)
            max_sim, _ = sims.max(dim=1)
            max_sim = max_sim.cpu().numpy()

            # label maybe tensor or python list
            if torch.is_tensor(label):
                lab_arr = label.cpu().numpy()
            else:
                lab_arr = np.array(label)

            for sim, lbl in zip(max_sim, lab_arr):
                if int(lbl) == 1:
                    pos_sims.append(float(sim))
                else:
                    neg_sims.append(float(sim))

    return pos_sims, neg_sims


# =========================
# Training loop with EMA + early stopping
# =========================
def train_with_early_stopping(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_cwes,
    patience=5,
    min_delta=1e-4,
    max_epochs=50,
    lambda_center=1.0,
    ema_momentum=0.9,
):
    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0
    epoch = 0
    train_loss_history = []
    val_loss_history = []
    hidden = model.encoder.config.hidden_size
    ema_centers = torch.zeros(num_cwes + 1, hidden, device=device)
    ema_initialized = False

    print("===== Start training =====")
    while epoch < max_epochs and patience_counter < patience:
        epoch += 1
        model.train()
        total_train_loss = 0.0
        train_count = 0

        for enc, labels, cwe_ids in tqdm(train_loader, desc=f"Training Epoch {epoch}"):
            enc = fix_batch_encoding(enc)
            enc = {k: v.to(device) for k, v in enc.items()}
            # labels_tensor = labels.detach().clone().long().to(device)
            cwe_ids_tensor = cwe_ids.detach().clone().long().to(device)

            logits, CLS = model(enc, cwe_ids_tensor)
            targets = cwe_ids_tensor.clone()
            targets[targets == -1] = num_cwes
            loss = F.cross_entropy(logits, targets)

            sep_loss = center_separation_loss(
                F.normalize(model.head.weight, dim=-1), margin=0.0
            )
            loss = loss + lambda_center * 0.1 * sep_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # EMA update
            with torch.no_grad():
                cls_norm = F.normalize(CLS, p=2, dim=-1)
                unique_labels = torch.unique(targets)
                batch_centers = torch.zeros_like(ema_centers)
                batch_counts = torch.zeros(num_cwes + 1, device=device)
                for lbl in unique_labels:
                    mask = targets == lbl
                    if mask.sum() == 0:
                        continue
                    mean_vec = cls_norm[mask].mean(dim=0)
                    batch_centers[lbl] = mean_vec
                    batch_counts[lbl] = mask.sum().float()
                if not ema_initialized:
                    for c in unique_labels:
                        ema_centers[c] = batch_centers[c]
                    if (batch_counts > 0).any():
                        ema_initialized = True
                else:
                    for c in unique_labels:
                        if batch_counts[c] > 0:
                            ema_centers[c] = (
                                ema_momentum * ema_centers[c]
                                + (1 - ema_momentum) * batch_centers[c]
                            )
                            ema_centers[c] = F.normalize(ema_centers[c], p=2, dim=-1)

            total_train_loss += loss.item()
            train_count += 1

        avg_train_loss = total_train_loss / max(1, train_count)
        train_loss_history.append(avg_train_loss)
        # val_loss 使用正+负样本
        avg_val_loss = compute_val_loss(model, val_loader, device)
        val_loss_history.append(avg_val_loss)

        if ema_initialized:
            pos_sims, neg_sims = compute_val_cosine_distribution(
                model, ema_centers, val_loader, device
            )
            pos_mean = np.mean(pos_sims) if len(pos_sims) > 0 else float("nan")
            neg_mean = np.mean(neg_sims) if len(neg_sims) > 0 else float("nan")
        else:
            pos_mean = float("nan")
            neg_mean = float("nan")

        print(
            f"[Epoch {epoch}] TrainLoss={avg_train_loss:.4f} ValLoss={avg_val_loss:.4f} | Pos sims mean={pos_mean:.4f}, Neg sims mean={neg_mean:.4f}"
        )

        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            best_model_state = {k: v for k, v in model.state_dict().items()}
            best_optimizer_state = {k: v for k, v in optimizer.state_dict().items()}
            patience_counter = 0
            best_epoch = epoch
            print(f"Validation loss improved to {avg_val_loss:.4f}, saving best model")
        else:
            patience_counter += 1
            print(f"No improvement. Patience {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(
            f"Loaded best model from epoch {best_epoch}, val_loss {best_val_loss:.4f}"
        )

    cls_train, labels_train = cache_cls_in_memory(model, train_loader, device)
    final_centers = compute_centers_from_cache(
        cls_train, labels_train, num_cwes, device=device
    )
    return (
        best_model_state,
        best_optimizer_state,
        train_loss_history,
        val_loss_history,
        best_epoch,
        best_val_loss,
        final_centers,
    )


# =========================
# 其余函数保持不变（compute_val_loss, search_tau, evaluate_open_set, visualize_tsne）
# =========================


def evaluate_open_set(model, cwe_centers, loader, tau, device):
    y_true, y_pred = [], []

    for enc, label, cwe_id in loader:
        preds, _ = open_set_predict(model, enc, cwe_centers, tau, device)

        # 标签
        if torch.is_tensor(label):
            lab_arr = label.cpu().numpy()
        else:
            lab_arr = np.array(label)

        if np.ndim(lab_arr) == 0:
            y_true.append(int(lab_arr))
        else:
            y_true.extend([int(x) for x in lab_arr])

        y_pred.extend(list(preds))

    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    tn = sum((yt == 0 and yp == 0) for yt, yp in zip(y_true, y_pred))
    fp = sum((yt == 0 and yp == 1) for yt, yp in zip(y_true, y_pred))
    spec = tn / (tn + fp + 1e-6)
    bacc = (spec + r) / 2

    return p, r, f1, acc, spec, bacc


def search_tau(model, cwe_centers, val_loader, device):
    # compute all max-sims once
    model.eval()
    all_max_sims = []
    all_labels = []

    # K E Y   C H A N G E : Exclude the last center (index num_cwes), which is the negative class center.
    positive_centers = cwe_centers[:-1].to(device)

    with torch.no_grad():
        for enc, label, cwe_id in val_loader:
            enc = fix_batch_encoding(enc)
            enc = {k: v.to(device) for k, v in enc.items()}
            vec = model.encode(enc)  # (B, H)

            # Use ONLY the positive class centers for open-set similarity
            sims = torch.matmul(vec, positive_centers.t())  # (B, num_cwes)

            max_sim, _ = sims.max(dim=1)
            all_max_sims.extend(max_sim.cpu().numpy().tolist())

            if torch.is_tensor(label):
                lab = label.cpu().numpy()
                if np.ndim(lab) == 0:
                    all_labels.append(int(lab))
                else:
                    all_labels.extend([int(x) for x in lab])
            else:
                all_labels.append(int(label))

    all_max_sims = np.array(all_max_sims)
    all_labels = np.array(all_labels)

    taus = np.linspace(all_max_sims.min() - 1e-6, all_max_sims.max() + 1e-6, 200)
    best_f1 = -1.0
    best_tau = 0.0
    for tau in taus:
        preds = (all_max_sims >= tau).astype(int)
        p = ((all_labels == 1) & (preds == 1)).sum()
        fp = ((all_labels == 0) & (preds == 1)).sum()
        fn = ((all_labels == 1) & (preds == 0)).sum()
        prec = p / (p + fp + 1e-8)
        rec = p / (p + fn + 1e-8)
        if prec + rec == 0:
            f1 = 0.0
        else:
            f1 = 2 * prec * rec / (prec + rec)
        if f1 > best_f1:
            best_f1 = f1
            best_tau = tau

    return best_tau, best_f1


def visualize_tsne(model, loader, device, fig_name="tsne.png"):
    model.eval()
    X_list, Y_list = [], []

    with torch.no_grad():
        for enc, label, cwe_id in loader:
            enc = fix_batch_encoding(enc)
            enc = {k: v.to(device) for k, v in enc.items()}
            CLS = model.encode(enc)  # (B, H)
            X_list.append(CLS.cpu().numpy())
            if torch.is_tensor(cwe_id):
                lab = cwe_id.cpu().numpy()
                if np.ndim(lab) == 0:
                    Y_list.append(int(lab))
                else:
                    Y_list.extend([int(x) for x in lab])
            else:
                Y_list.append(int(cwe_id))

    X = np.concatenate(X_list, axis=0)
    Y = np.array(Y_list)

    tsne = TSNE(n_components=2, perplexity=40, learning_rate="auto", init="pca")
    X_2d = tsne.fit_transform(X)

    plt.figure(figsize=(8, 8))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=Y, s=4, alpha=0.6)
    plt.title("t-SNE of Validation Set")
    plt.savefig(fig_name)
    plt.close()


# =========================
# Main
# =========================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = RobertaTokenizer.from_pretrained("codemetic/CweBERT-mlm")
    raw = load_dataset("codemetic/curve", DATASET, split=SPLIT)

    rng = np.random.default_rng(42)

    # -----------------
    # 分离正负样本
    # -----------------
    pos_indices = [i for i, item in enumerate(raw) if int(item["label"]) == 1]
    neg_indices = [i for i, item in enumerate(raw) if int(item["label"]) == 0]

    # 分层划分正样本（每类至少一个验证样本）
    grouped = defaultdict(list)
    for idx in pos_indices:
        cwe = raw[idx]["cwe"]
        grouped[cwe].append(idx)

    train_pos, val_pos, test_pos = [], [], []
    for cwe, indices in grouped.items():
        indices = np.array(indices)
        rng.shuffle(indices)
        n = len(indices)
        n_val = max(1, int(n * 0.1))
        n_train = n - n_val - max(1, int(n * 0.1))
        n_test = n - n_train - n_val
        train_pos.extend(indices[:n_train])
        val_pos.extend(indices[n_train : n_train + n_val])
        test_pos.extend(indices[n_train + n_val :])

    # 负样本按比例划分
    neg_indices = np.array(neg_indices)
    rng.shuffle(neg_indices)
    n_neg = len(neg_indices)
    n_train_neg = int(n_neg * 0.8)
    n_val_neg = int(n_neg * 0.1)
    train_neg = neg_indices[:n_train_neg].tolist()
    val_neg = neg_indices[n_train_neg : n_train_neg + n_val_neg].tolist()
    test_neg = neg_indices[n_train_neg + n_val_neg :].tolist()

    # 合并
    train_ids = train_pos + train_neg
    val_ids = val_pos + val_neg
    test_ids = test_pos + test_neg
    rng.shuffle(train_ids)
    rng.shuffle(val_ids)
    rng.shuffle(test_ids)

    hf = DatasetDict(
        {
            "train": raw.select(train_ids),
            "validation": raw.select(val_ids),
            "test": raw.select(test_ids),
        }
    )
    print(
        f"[Split Done] Train={len(hf['train'])}, Val={len(hf['validation'])}, Test={len(hf['test'])}"
    )

    # CWE → ID
    cwe_counts = {}
    for item in hf["train"]:
        if int(item["label"]) == 1:
            c = item["cwe"]
            cwe_counts[c] = cwe_counts.get(c, 0) + 1
    cwe2id = {c: i for i, c in enumerate(sorted(cwe_counts.keys()))}
    num_cwes = len(cwe2id)
    print(f"[CWE] num_cwes = {num_cwes}")
    margins = build_adaptive_margins(cwe_counts).to(device)

    train_set = CweDataset(hf["train"], tokenizer, cwe2id, is_train=True)
    val_set = CweDataset(hf["validation"], tokenizer, cwe2id, is_train=False)
    test_set = CweDataset(hf["test"], tokenizer, cwe2id, is_train=False)

    train_loader = DataLoader(
        train_set,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    model = CweBertArcFace(
        model_name="codemetic/CweBERT-mlm", num_cwes=num_cwes, adaptive_margins=margins
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    (
        best_model_state,
        best_optimizer_state,
        train_loss_history,
        val_loss_history,
        best_epoch,
        best_val_loss,
        cwe_centers,
    ) = train_with_early_stopping(
        model, train_loader, val_loader, optimizer, device, num_cwes, patience=3
    )

    # 绘图
    plt.figure(figsize=(12, 6))
    plt.plot(train_loss_history, label="Train Loss")
    plt.plot(val_loss_history, label="Val Loss", color="orange")
    plt.axvline(
        x=best_epoch - 1, color="red", linestyle="--", label=f"Best Epoch {best_epoch}"
    )
    plt.grid(True)
    plt.legend()
    plt.savefig(f"training_validation_loss-{DATASET}.png")
    plt.close()

    # Open-set threshold
    cls_val, labels_val = cache_cls_in_memory(model, val_loader, device)
    cwe_centers_tensor = compute_centers_from_cache(
        cls_val, labels_val, num_cwes, device=device
    )
    best_tau, best_f1 = search_tau(model, cwe_centers_tensor, val_loader, device)
    p, r, f1, acc, sp, bacc = evaluate_open_set(
        model, cwe_centers, test_loader, best_tau, device
    )
    print(f"[Open-set] Tau={best_tau:.4f},best val loss={best_val_loss}")
    print("===== Final Test Result =====")
    print(f"Precision={p:.4f} Recall={r:.4f} F1={f1:.4f}")
    print(f"Accuracy={acc:.4f} Specificity={sp:.4f} BalancedAcc={bacc:.4f}")

    visualize_tsne(model, test_loader, device, f"tsne_test_{DATASET}.png")

    # 保存模型
    save_dir = "./saved_model"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_model_state,
            "optimizer_state_dict": best_optimizer_state,
            "cwe_centers": cwe_centers_tensor,
            "cwe2id": cwe2id,
            "tau": best_tau,
        },
        os.path.join(save_dir, f"cwebert_{DATASET}.pt"),
    )
    print(f"Model saved to {save_dir}")


if __name__ == "__main__":
    main()
