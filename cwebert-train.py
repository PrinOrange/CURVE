#!/usr/bin/env python3
# train.py
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

# -------------------------
# small helper to fix batched encodings from tokenizer
# When DataLoader batches items whose tokenizer returned tensors shape (1, L),
# collated tensor becomes (B, 1, L). We want (B, L).
# This helper leaves already-correct tensors unchanged.
# -------------------------
def fix_batch_encoding(enc):
    out = {}
    for k, v in enc.items():
        if torch.is_tensor(v):
            # typical cases:
            # (B, L) -> keep
            # (B, 1, L) -> squeeze(1) -> (B, L)
            # (1, L) -> keep (single-sample)
            # (1, 1, L) -> squeeze(1) -> (1, L)
            if v.dim() == 3 and v.size(1) == 1:
                out[k] = v.squeeze(1)
            else:
                out[k] = v
        else:
            out[k] = v
    return out


# ============================================================================`
# 1. Dataset
# ============================================================================


# =========================
# Dataset (训练只返回正样本)
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
                if label != 1:
                    continue
                cwe = item["cwe"]  # assume single CWE
                if cwe not in cwe2id:
                    continue  # 忽略训练集没有的 CWE
                cwe_id = cwe2id[cwe]
                self.samples.append((code, cwe_id))
            else:
                # val/test: 正样本可能是未知 CWE
                cwe = item["cwe"] if label == 1 and item.get("cwe", None) else None
                cwe_id = cwe2id[cwe] if (cwe in cwe2id) else -1
                self.samples.append((code, label, cwe_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.is_train:
            code, cwe_id = self.samples[idx]
            enc = self.tokenizer(
                code,
                max_length=512,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            return enc, cwe_id
        else:
            code, label, cwe_id = self.samples[idx]
            enc = self.tokenizer(
                code,
                max_length=512,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            return enc, label, cwe_id


# ============================================================================`
# 2. Adaptive-Margin ArcFace head (keeps your adaptive margins idea)
# ============================================================================


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


# ============================================================================`
# 3. Main model (CweBERT-mlm backbone + ArcFace)
# ============================================================================


class CweBertArcFace(nn.Module):
    def __init__(self, model_name, num_cwes, adaptive_margins):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained(model_name)
        H = self.encoder.config.hidden_size
        self.head = ArcFace(H, num_cwes, adaptive_margins)

    def forward(self, encodings, labels):
        # encodings: dict with tensors (B, L)
        out = self.encoder(**encodings)
        CLS = out.last_hidden_state[:, 0, :]  # [B, H]
        logits = self.head(CLS, labels)
        return logits, CLS

    def encode(self, encodings):
        out = self.encoder(**encodings)
        CLS = out.last_hidden_state[:, 0, :]
        return F.normalize(CLS, p=2, dim=-1)


# ============================================================================`
# 4. Adaptive Margin (frequency-based) - 修改为稍微更大一些
# ============================================================================


def build_adaptive_margins(cwe_counts):
    # make margins larger than original tiny values
    margins = []
    for cwe, n in cwe_counts.items():
        # larger margin for low-frequency classes
        m = 0.35 + 0.5 / math.sqrt(n + 1)  # baseline + frequency term
        margins.append(m)
    return torch.tensor(margins, dtype=torch.float32)


# ============================================================================`
# 5. Open-set classification (prediction helper)
# ============================================================================


def open_set_predict(model, encodings, cwe_centers, tau, device):
    model.eval()
    with torch.no_grad():
        encodings = fix_batch_encoding(encodings)
        encodings = {k: v.to(device) for k, v in encodings.items()}
        vec = model.encode(encodings)  # (B,H)
        centers = cwe_centers.to(vec.device)
        sims = torch.matmul(vec, centers.t())  # (B, C)
        max_sim, argmax = sims.max(dim=1)
        preds = (max_sim >= tau).long().cpu().numpy()  # 1: vuln, 0: non-vuln
        return preds, argmax.cpu().numpy()


# ============================================================================`
# 6. Cache CLS in memory (used only when we want final/precise centers)
# ============================================================================


def cache_cls_in_memory(model, loader, device):
    print("===== Caching CLS to memory =====")
    model.eval()

    all_cls = []
    all_labels = []

    with torch.no_grad():
        for enc, labels in tqdm(loader, desc="Caching CLS"):
            enc = fix_batch_encoding(enc)
            # move enc to device (squeezing any (B,1,L) dims)
            enc = {k: v.squeeze(1).to(device) for k, v in enc.items()}
            # fetch CLS
            cls = model.encode(enc)  # shape: [B, hidden], normalized
            # store on cpu to save GPU RAM
            all_cls.append(cls.cpu())
            # labels may be list/np/scalar
            if torch.is_tensor(labels):
                all_labels.append(labels.cpu())
            else:
                # labels from CweDataset for train are ints or list
                all_labels.append(torch.tensor(labels, dtype=torch.long))

    all_cls = torch.cat(all_cls, dim=0)  # [N, hidden]
    all_labels = torch.cat(all_labels, dim=0)

    print(f"Cached {len(all_cls)} CLS vectors in memory.")
    return all_cls, all_labels


def compute_centers_from_cache(cls_cached, labels_cached, num_cwes, device=None):
    """
    cls_cached: tensor [N, D] (may be on cpu)
    labels_cached: tensor [N]  (may be on cpu)
    returns: centers tensor [num_cwes, D] placed on `device` (if provided) otherwise same device as cls_cached
    """
    if device is None:
        device = cls_cached.device

    hidden = cls_cached.size(1)
    centers = torch.zeros(num_cwes, hidden, device=device)

    labels_cached = labels_cached.to(device)
    cls_cached = cls_cached.to(device)

    for cid in range(num_cwes):
        mask = labels_cached == cid
        if mask.sum() == 0:
            # keep zero (will be normalized to zero vector)
            continue
        vec = cls_cached[mask]
        centers[cid] = F.normalize(vec.mean(dim=0), p=2, dim=-1)

    return centers


# ============================================================================`
# 7. Evaluate and utilities
# ============================================================================


def compute_val_cosine_distribution(model, cwe_centers, val_loader, device):
    """Return list of cosine similarities (max_sim) for positive & negative samples"""
    model.eval()
    pos_sims = []
    neg_sims = []
    centers_device = cwe_centers.to(device)

    with torch.no_grad():
        for enc, label, cwe_id in val_loader:
            enc = fix_batch_encoding(enc)
            enc = {k: v.to(device) for k, v in enc.items()}
            vec = model.encode(enc)  # (B, H)

            sims = torch.matmul(vec, centers_device.t())  # (B, C)
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

    with torch.no_grad():
        for enc, label, cwe_id in val_loader:
            enc = fix_batch_encoding(enc)
            enc = {k: v.to(device) for k, v in enc.items()}
            vec = model.encode(enc)  # (B, H)
            sims = torch.matmul(vec, cwe_centers.to(device).t())  # (B, C)
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

            cwe_ids_vuln = [cwe_id[i] for i in range(len(lab_arr)) if int(lab_arr[i]) == 1]
            if len(cwe_ids_vuln) == 0:
                continue
            labels_vuln = torch.tensor(cwe_ids_vuln, dtype=torch.long, device=device)

            logits, _ = model(enc_vuln, labels_vuln)
            loss = F.cross_entropy(logits, labels_vuln)

            total_loss += loss.item()
            count += 1

    return total_loss / max(1, count)


def center_separation_loss(centers, margin=0.0):
    """
    centers: (C, D) normalized
    margin: desired max allowed cosine between centers (e.g., 0.0 means orthogonal)
    returns scalar loss: mean squared of positive (sim - margin)
    """
    sim = torch.matmul(centers, centers.t())  # (C, C)
    C = sim.shape[0]
    mask = ~torch.eye(C, dtype=torch.bool, device=sim.device)
    sims_offdiag = sim[mask]  # (C*C-C,)
    slack = torch.clamp(sims_offdiag - margin, min=0.0)
    loss = (slack**2).mean()
    return loss


# ============================================================================`
# 10. Main training with EMA centers + early stopping
# ============================================================================
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
    """
    Returns:
      best_model (state_dict),
      train_loss_history,
      val_loss_history,
      best_epoch,
      best_val_loss,
      final_centers (on device)
    """
    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0
    epoch = 0

    train_loss_history = []
    val_loss_history = []

    # initialize EMA centers on device
    hidden = model.encoder.config.hidden_size
    ema_centers = torch.zeros(num_cwes, hidden, device=device)
    ema_initialized = False

    print("===== Start training =====")
    while epoch < max_epochs and patience_counter < patience:
        epoch += 1
        model.train()
        total_train_loss = 0.0
        train_count = 0

        # ---------- 训练 ----------
        for enc, labels in tqdm(train_loader, desc=f"Training Epoch {epoch}"):
            enc = fix_batch_encoding(enc)
            enc = {k: v.to(device) for k, v in enc.items()}
            # labels in training are cwe_ids (ints)
            if torch.is_tensor(labels):
                labels_tensor = labels.detach().clone().long().to(device)
            else:
                labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)

            logits, CLS = model(enc, labels_tensor)
            loss = F.cross_entropy(logits, labels_tensor)

            # center separation on model.head.weight (helps prevent collapse)
            sep_loss = center_separation_loss(F.normalize(model.head.weight, dim=-1), margin=0.0)
            loss = loss + lambda_center * 0.1 * sep_loss  # small weight for separation

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update EMA centers using this batch (only for classes present in batch)
            with torch.no_grad():
                cls_norm = F.normalize(CLS, p=2, dim=-1)  # (B, H)
                # compute per-class batch mean
                unique_labels = torch.unique(labels_tensor)
                batch_centers = torch.zeros_like(ema_centers)
                batch_counts = torch.zeros(num_cwes, device=device)
                for lbl in unique_labels:
                    mask = labels_tensor == lbl
                    if mask.sum() == 0:
                        continue
                    mean_vec = cls_norm[mask].mean(dim=0)
                    batch_centers[lbl] = mean_vec
                    batch_counts[lbl] = mask.sum().float()

                # if not initialized, initialize those classes in ema_centers
                if not ema_initialized:
                    # initialize only classes that appear in this batch
                    for c in unique_labels:
                        ema_centers[c] = batch_centers[c]
                    # check if there is any non-zero centers
                    if (batch_counts > 0).any():
                        ema_initialized = True
                else:
                    # update using momentum for classes that have counts
                    for c in unique_labels:
                        if batch_counts[c] > 0:
                            ema_centers[c] = ema_momentum * ema_centers[c] + (1 - ema_momentum) * batch_centers[c]
                            ema_centers[c] = F.normalize(ema_centers[c], p=2, dim=-1)

            total_train_loss += loss.item()
            train_count += 1

        avg_train_loss = total_train_loss / max(1, train_count)
        train_loss_history.append(avg_train_loss)

        # ---------- 验证 ----------
        avg_val_loss = compute_val_loss(model, val_loader, device)
        val_loss_history.append(avg_val_loss)

        # compute pos/neg sims using current ema_centers (fallback: if not initialized, skip)
        if ema_initialized:
            pos_sims, neg_sims = compute_val_cosine_distribution(model, ema_centers, val_loader, device)
            pos_mean = np.mean(pos_sims) if len(pos_sims) > 0 else float("nan")
            neg_mean = np.mean(neg_sims) if len(neg_sims) > 0 else float("nan")
        else:
            pos_mean = float("nan")
            neg_mean = float("nan")

        print(
            f"[Epoch {epoch}] TrainLoss={avg_train_loss:.4f} ValLoss={avg_val_loss:.4f} | Pos sims mean={pos_mean:.4f}, Neg sims mean={neg_mean:.4f}"
        )

        # Early stopping (based on val_loss)
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            best_epoch = epoch
            print(f"Validation loss improved to {avg_val_loss:.4f}, saving best model")
        else:
            patience_counter += 1
            print(f"No improvement. Patience {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    # Load best model state into model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model from epoch {best_epoch}, val_loss {best_val_loss:.4f}")

    # After training: compute precise final centers by caching CLS once (on best model)
    cls_train, labels_train = cache_cls_in_memory(model, train_loader, device)
    final_centers = compute_centers_from_cache(cls_train, labels_train, num_cwes, device=device)

    return best_model_state, train_loss_history, val_loss_history, best_epoch, best_val_loss, final_centers


# ============================================================================`
# 11. t-SNE visualization
# ============================================================================


def visualize_tsne(model, cwe_centers, loader, device, fig_name="tsne.png"):
    model.eval()
    X_list, Y_list = [], []

    with torch.no_grad():
        for enc, label, cwe_id in loader:
            enc = fix_batch_encoding(enc)
            enc = {k: v.to(device) for k, v in enc.items()}
            CLS = model.encode(enc)  # (B, H)
            X_list.append(CLS.cpu().numpy())
            if torch.is_tensor(label):
                lab = label.cpu().numpy()
                if np.ndim(lab) == 0:
                    Y_list.append(int(lab))
                else:
                    Y_list.extend([int(x) for x in lab])
            else:
                Y_list.append(int(label))

    X = np.concatenate(X_list, axis=0)
    Y = np.array(Y_list)

    tsne = TSNE(n_components=2, perplexity=40, learning_rate="auto", init="pca")
    X_2d = tsne.fit_transform(X)

    plt.figure(figsize=(8, 8))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=Y, s=4, alpha=0.6)
    plt.title("t-SNE of Validation Set")
    plt.savefig(fig_name)
    plt.close()


# ============================================================================`
# 12. Everything together (main)
# ============================================================================


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = RobertaTokenizer.from_pretrained("codemetic/CweBERT-mlm")

    # -------------------------
    # 1. Load HF dataset
    # -------------------------
    raw = load_dataset("codemetic/curve", "diversevul")["data"]

    # -------------------------
    # 2. Extract CWE for each item
    # -------------------------
    def get_cwe(item):
        if int(item["label"]) == 1 and item.get("cwe", None):
            return item["cwe"]
        return None

    all_cwes = [get_cwe(item) for item in raw]

    # -------------------------
    # 3. Group indices by CWE (only positive samples)
    # -------------------------
    grouped = defaultdict(list)
    for idx, cwe in enumerate(all_cwes):
        if cwe is not None:
            grouped[cwe].append(idx)

    # -------------------------
    # 4. Stratified split: 80% train, 10% val, 10% test
    # -------------------------
    rng = np.random.default_rng(42)
    train_ids, val_ids, test_ids = [], [], []

    # positive grouped split (as before)
    for cwe, indices in grouped.items():
        indices = np.array(indices)
        rng.shuffle(indices)

        n = len(indices)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)

        train_ids.extend(indices[:n_train])
        val_ids.extend(indices[n_train : n_train + n_val])
        test_ids.extend(indices[n_train + n_val :])

    # collect negative indices (label == 0 OR label==1 but no cwe)
    all_indices = np.arange(len(raw))
    positive_mask = np.zeros(len(raw), dtype=bool)
    for idx_list in grouped.values():
        positive_mask[idx_list] = True
    neg_indices = all_indices[~positive_mask]

    # shuffle negatives and split them into validation & test (e.g., 50/50)
    rng.shuffle(neg_indices)
    if len(neg_indices) > 0:
        n_neg_val = int(0.5 * len(neg_indices))
        neg_val = neg_indices[:n_neg_val]
        neg_test = neg_indices[n_neg_val:]
        # add negatives to val/test (not to train)
        val_ids.extend(neg_val.tolist())
        test_ids.extend(neg_test.tolist())

    # optional: shuffle final splits
    rng.shuffle(train_ids)
    rng.shuffle(val_ids)
    rng.shuffle(test_ids)

    # Convert into HF splits
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

    # -------------------------
    # 5. Build CWE → ID mapping (from TRAIN only)
    # -------------------------
    cwe_counts = {}
    for item in hf["train"]:
        if int(item["label"]) == 1:
            c = item["cwe"]
            cwe_counts[c] = cwe_counts.get(c, 0) + 1

    cwe2id = {c: i for i, c in enumerate(sorted(cwe_counts.keys()))}
    num_cwes = len(cwe2id)
    print(f"[CWE] num_cwes = {num_cwes}")

    # adaptive margins
    margins = build_adaptive_margins(cwe_counts).to(device)

    # -------------------------
    # 6. Datasets
    # -------------------------
    train_set = CweDataset(hf["train"], tokenizer, cwe2id, is_train=True)
    val_set = CweDataset(hf["validation"], tokenizer, cwe2id, is_train=False)
    test_set = CweDataset(hf["test"], tokenizer, cwe2id, is_train=False)

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=8, shuffle=False)

    # -------------------------
    # 7. Build model
    # -------------------------
    model = CweBertArcFace(
        model_name="codemetic/CweBERT-mlm",
        num_cwes=num_cwes,
        adaptive_margins=margins,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    # -------------------------
    # 8. Train with early stopping
    # -------------------------
    print("Starting training with early stopping...")
    (
        best_model_state,
        train_loss_history,
        val_loss_history,
        best_epoch,
        best_val_loss,
        cwe_centers,
    ) = train_with_early_stopping(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_cwes,
        patience=3,
        min_delta=1e-4,
        max_epochs=50,
        lambda_center=1.0,
        ema_momentum=0.9,
    )

    # plot losses
    plt.figure(figsize=(12, 6))
    plt.plot(train_loss_history, label="Train Loss")
    plt.plot(val_loss_history, label="Val Loss", color="orange")
    plt.axvline(
        x=best_epoch - 1, color="red", linestyle="--", label=f"Best Epoch {best_epoch}"
    )
    plt.grid(True)
    plt.legend()
    plt.savefig("training_validation_loss.png")
    plt.close()

    # -------------------------
    # 9. (cwe_centers already computed at end of training)
    # -------------------------
    print("Final centers computed and stored in variable cwe_centers (on device).")

    # -------------------------
    # 10. Search best open-set threshold tau
    # -------------------------
    best_tau, best_f1 = search_tau(model, cwe_centers, val_loader, device)
    print("Best tau =", best_tau, "F1=", best_f1)

    # -------------------------
    # 11. Final evaluation
    # -------------------------
    p, r, f1, acc, sp, bacc = evaluate_open_set(
        model, cwe_centers, test_loader, best_tau, device
    )
    print("===== Final Test Result =====")
    print(f"Precision={p:.4f} Recall={r:.4f} F1={f1:.4f}")
    print(f"Accuracy={acc:.4f} Specificity={sp:.4f} BalancedAcc={bacc:.4f}")

    # -------------------------
    # 12. t-SNE visualization
    # -------------------------
    visualize_tsne(model, cwe_centers, test_loader, device, "tsne_test.png")

    # -------------------------
    # 13. Save best model
    # -------------------------
    torch.save(
        {
            "model_state_dict": best_model_state,
            "cwe_centers": cwe_centers.cpu(),
            "best_tau": best_tau,
            "cwe2id": cwe2id,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
        },
        "best_model.pth",
    )

    print(
        f"Best model saved to best_model.pth (epoch {best_epoch}, val_loss {best_val_loss:.4f})"
    )


if __name__ == "__main__":
    main()
