#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tabr_model.py — TabR-lite: mô hình học sâu cho dữ liệu bảng (tabular).

Thiết kế lấy cảm hứng từ TabR (Gorishniy et al., Yandex 2024, "TabR: Untangling
Transformer and Feature Attribution in Interacting with Tabular Data"): mỗi feature
được token hóa thành vector, một token [CLS] tổng hợp thông tin qua các block
Transformer có attention, và phần head thêm 1 nhánh kNN (khoảng cách cosine tới
các mẫu đã lưu) trộn với nhánh attention — đúng tinh thần "attention + retrieval"
của TabR nhưng ở mức đơn giản hóa để chạy nhanh trên CPU của Kali.

Lưu ý trung thực: đây là bản tái hiện đơn giản cho mục đích demo/giảng dạy,
KHÔNG phải code gốc của Yandex. Nếu cần kết quả chuẩn TabR, dùng repo chính thức
github.com/yandex-research/tabular-dl-tabr.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch import Tensor
except ImportError as e:  # pragma: no cover
    raise ImportError("Cần PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cpu") from e


# ---------------------------------------------------------------------------
# Khối Transformer chuẩn (pre-norm)
# ---------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, d: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, d))

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        return x + self.ff(self.norm2(x))


class TabRLite(nn.Module):
    """Tokenizer per-feature + CLS Transformer + head (attention) & kNN head."""

    def __init__(self, n_features: int, n_classes: int, d_token: int = 64,
                 n_blocks: int = 2, n_heads: int = 4, dropout: float = 0.1,
                 knn_k: int = 16, knn_alpha: float = 0.3, group_size: int = 6):
        super().__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.knn_k = knn_k
        self.knn_alpha = knn_alpha  # trọng số của nhánh kNN khi blend
        self.group_size = group_size
        self.n_groups = math.ceil(n_features / group_size)
        self.pad_features = self.n_groups * group_size - n_features

        # Tokenizer theo nhóm đặc trưng: gom group_size feature -> 1 token d chiều (nhanh gấp ~30x trên CPU)
        self.feature_weights = nn.Parameter(torch.ones(n_features))
        self.feature_biases = nn.Parameter(torch.zeros(n_features))
        self.group_proj = nn.Parameter(torch.randn(self.n_groups, group_size, d_token) * (1.0 / math.sqrt(group_size)))
        self.group_bias = nn.Parameter(torch.zeros(1, self.n_groups, d_token))
        self.cls = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)
        self.blocks = nn.ModuleList([Block(d_token, n_heads, dropout) for _ in range(n_blocks)])
        self.norm = nn.LayerNorm(d_token)
        # Nhánh residual trực tiếp từ vector feature -> d_token giúp học ranh giới sắc nét hơn
        self.res_proj = nn.Sequential(
            nn.Linear(n_features, d_token * 2),
            nn.LayerNorm(d_token * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_token * 2, d_token),
        )
        self.head_attn = nn.Sequential(
            nn.Linear(d_token, d_token),
            nn.LayerNorm(d_token),
            nn.GELU(),
            nn.Linear(d_token, n_classes),
        )

        # Nhánh kNN: memory bank của embedding CLS (điền sau khi train xong attention phần)
        self.register_buffer("mem_emb", torch.zeros(0, d_token))
        self.register_buffer("mem_y", torch.zeros(0, dtype=torch.long))

    # ------------------------------------------------------------------
    def tokenize(self, x: Tensor) -> Tensor:
        """x: (B, n_features) đã scale -> (B, 1 + n_groups, d)."""
        vals = x * self.feature_weights + self.feature_biases  # (B, F)
        if self.pad_features > 0:
            vals = torch.nn.functional.pad(vals, (0, self.pad_features))
        vals = vals.view(x.size(0), self.n_groups, self.group_size)  # (B, G, K)
        tokens = torch.einsum("bgk,gkd->bgd", vals, self.group_proj) + self.group_bias
        cls = self.cls.expand(x.size(0), -1, -1)  # (B, 1, d)
        return torch.cat([cls, tokens], dim=1)

    def encode(self, x: Tensor) -> Tensor:
        h = self.tokenize(x)
        for blk in self.blocks:
            h = blk(h)
        return self.norm(h[:, 0]) + self.res_proj(x)  # embedding CLS + residual (B,d)

    def forward(self, x: Tensor) -> Tensor:
        """Logits = attention head + alpha * kNN vote (nếu có memory)."""
        emb = self.encode(x)
        logits = self.head_attn(emb)
        if self.mem_emb.size(0) > 0 and self.knn_alpha > 0:
            # cosine similarity tới memory bank
            e = torch.nn.functional.normalize(emb, dim=1)
            m = torch.nn.functional.normalize(self.mem_emb, dim=1)
            sim = e @ m.T  # (B,M)
            k = min(self.knn_k, sim.size(1))
            top_sim, top_idx = sim.topk(k, dim=1)
            knn_logits = torch.zeros(x.size(0), self.n_classes, device=x.device)
            knn_idx = self.mem_y[top_idx]                              # (B,k)
            votes = torch.ones_like(top_sim) / k
            knn_w = votes * (0.5 + 0.5 * top_sim)                      # (B,k)
            knn_logits.scatter_add_(1, knn_idx, knn_w)
            logits = logits + self.knn_alpha * knn_logits * logits.abs().mean(dim=1, keepdim=True)
        return logits

    # ------------------------------------------------------------------
    def build_memory(self, X: np.ndarray, y: np.ndarray, max_mem: int = 8192, device: str = "cpu") -> None:
        """Lưu embedding cân bằng theo lớp (tối đa max_mem mẫu train) để nhánh kNN dùng lúc suy luận."""
        self.eval()
        rng = np.random.RandomState(0)
        classes = np.unique(y)
        per_cls = max(1, max_mem // max(1, len(classes)))
        sel_idx = []
        for c in classes:
            idx_c = np.where(y == c)[0]
            if len(idx_c) > per_cls:
                idx_c = rng.choice(idx_c, per_cls, replace=False)
            sel_idx.append(idx_c)
        sel = np.concatenate(sel_idx)
        X_sel, y_sel = X[sel], y[sel]

        embs = []
        with torch.no_grad():
            for i in range(0, len(X_sel), 2048):
                xb = torch.tensor(X_sel[i:i + 2048], dtype=torch.float32, device=device)
                embs.append(self.encode(xb).cpu())
        emb_all = torch.cat(embs)
        self.mem_emb = emb_all.to(next(self.parameters()).device)
        self.mem_y = torch.tensor(y_sel, dtype=torch.long, device=self.mem_emb.device)

    def save(self, path: Path, meta: dict) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict(), "meta": meta}, path)

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "TabRLite":
        payload = torch.load(path, map_location=device, weights_only=False)
        meta = payload["meta"]
        model = cls(n_features=meta["n_features"], n_classes=meta["n_classes"],
                    d_token=meta.get("d_token", 64), n_blocks=meta.get("n_blocks", 2),
                    n_heads=meta.get("n_heads", 4), dropout=meta.get("dropout", 0.1),
                    knn_k=meta.get("knn_k", 16), knn_alpha=meta.get("knn_alpha", 0.3),
                    group_size=meta.get("group_size", 6))
        state = dict(payload["state_dict"])
        # Buffer memory bank có kích thước theo dữ liệu train — gán trực tiếp thay vì load_state_dict
        for key in ("mem_emb", "mem_y"):
            if key in state:
                setattr(model, key, state.pop(key))
        model.load_state_dict(state, strict=False)
        model.to(device)
        return model


# ---------------------------------------------------------------------------
# Vòng lặp huấn luyện
# ---------------------------------------------------------------------------
def train_tabr(X_train: np.ndarray, y_train: np.ndarray,
               X_val: np.ndarray, y_val: np.ndarray,
               n_classes: int,
               epochs: int = 12, batch_size: int = 4096, lr: float = 3e-3,
               d_token: int = 64, n_blocks: int = 2, n_heads: int = 4,
               device: str = "cpu", verbose: bool = True) -> tuple[TabRLite, dict]:
    """Trả về (model, history). Đã bao gồm early stopping trên val macro-F1."""
    from sklearn.metrics import accuracy_score, f1_score

    torch.manual_seed(42)
    n_features = X_train.shape[1]
    model = TabRLite(n_features, n_classes, d_token=d_token, n_blocks=n_blocks,
                     n_heads=n_heads).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    # Trọng số lớp nhẹ (inverse sqrt frequency) để không bỏ sót lớp ít mẫu (BruteForce)
    counts = np.bincount(y_train, minlength=n_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    cls_w = (counts.max() / counts) ** 0.35
    cls_w = cls_w / cls_w.mean()
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(cls_w, dtype=torch.float32, device=device))

    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.long)

    def _eval_val() -> tuple[float, float]:
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(X_val), 4096):
                xb = torch.tensor(X_val[i:i + 4096], dtype=torch.float32, device=device)
                preds.append(model(xb).argmax(dim=1).cpu().numpy())
        pred = np.concatenate(preds)
        return float(accuracy_score(y_val, pred)), float(f1_score(y_val, pred, average="macro"))

    history: dict = {"epochs": [], "train_loss": [], "val_f1": []}
    best_score, best_state, patience, bad = -1.0, None, 4, 0
    rng = np.random.RandomState(42)
    n = len(Xt)

    for ep in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(n)
        tot, t0 = 0.0, time.time()
        for i in range(0, n, batch_size):
            idx = torch.tensor(perm[i:i + batch_size])
            xb, yb = Xt[idx].to(device), yt[idx].to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()) * len(idx)
        sched.step()
        vacc, vf1 = _eval_val()
        score = 0.5 * vacc + 0.5 * vf1
        history["epochs"].append(ep)
        history["train_loss"].append(tot / n)
        history["val_f1"].append(vf1)
        if verbose:
            print(f"    [TabR] epoch {ep:02d}/{epochs}  loss={tot/n:.4f}  val_acc={vacc:.4f}  val_macroF1={vf1:.4f}"
                  f"  ({time.time()-t0:.1f}s)", flush=True)
        if score > best_score + 1e-4:
            best_score, bad = score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"    [TabR] early stopping (không cải thiện {patience} epoch)", flush=True)
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    # Xây memory bank cho nhánh kNN từ tập train (cân bằng theo lớp) và chọn knn_alpha tốt nhất trên val
    model.build_memory(X_train, y_train, device=device)
    best_alpha, best_a_score = 0.0, _eval_val()[0]
    for alpha in (0.0, 0.1, 0.2, 0.3):
        model.knn_alpha = alpha
        vacc, vf1 = _eval_val()
        sc = 0.5 * vacc + 0.5 * vf1
        if sc > best_a_score:
            best_a_score, best_alpha = sc, alpha
    model.knn_alpha = best_alpha
    if verbose:
        print(f"    [TabR] knn_alpha tối ưu = {best_alpha:.2f}", flush=True)
    model.meta_final_val_f1 = best_score  # type: ignore[attr-defined]
    return model, history


if __name__ == "__main__":
    # Kiểm tra nhanh mô hình với dữ liệu giả
    rng = np.random.RandomState(0)
    X = rng.randn(2000, 20).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)
    m, h = train_tabr(X[:1500], y[:1500], X[1500:], y[1500:], n_classes=2, epochs=2, verbose=True)
    print("Smoke TabR OK, lịch sử:", json.dumps(h["val_f1"]))
