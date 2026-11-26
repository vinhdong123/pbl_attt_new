#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eda.py — Phân tích khám phá (EDA) dataset CIC-IDS2017, sinh báo cáo cho GUI.

Kết quả ghi vào ./reports/:
  eda.json                        — số liệu thống kê (GUI đọc để hiển thị bảng)
  eda_class_distribution.png      — phân bố 15 nhãn gốc
  eda_grouped_distribution.png    — phân bố 5 lớp chiến thuật
  eda_flows_per_file.png          — benign/attack theo từng ngày (file)
  eda_feature_distributions.png   — histogram các feature quan trọng
  eda_correlation.png             — tương quan giữa các feature

Cách chạy:
    python3 eda.py                    # đọc đủ cột Label + lấy mẫu feature
    python3 eda.py --sample 100000    # lấy mẫu feature ít hơn (nhanh hơn)
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent

LABEL_MAP = {
    "BENIGN": "BENIGN",
    "DDoS": "DDoS", "PortScan": "PortScan",
    "DoS Hulk": "DoS", "DoS GoldenEye": "DoS", "DoS slowloris": "DoS", "DoS Slowhttptest": "DoS",
    "FTP-Patator": "BruteForce", "SSH-Patator": "BruteForce",
}
GROUP_ORDER = ["BENIGN", "DoS", "DDoS", "PortScan", "BruteForce"]
KEY_FEATURES = ["Destination Port", "Flow Duration", "Flow Bytes/s", "Flow Packets/s",
                "Total Fwd Packets", "Total Length of Fwd Packets", "Fwd IAT Mean",
                "Flow IAT Std", "SYN Flag Count", "Average Packet Size"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _label_column(chunk: pd.DataFrame) -> pd.Series:
    chunk.columns = [c.strip() for c in chunk.columns]
    if "Label" not in chunk.columns:
        raise ValueError("Không tìm thấy cột Label")
    return chunk["Label"].astype(str).str.strip()


def generate_eda(data_dir: Path, reports_dir: Path, sample_per_file: int = 200_000) -> dict:
    files = sorted(data_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"Không có CSV trong {data_dir} — chạy download_dataset.py")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ pass 1: đếm nhãn
    label_counts: Counter = Counter()
    row_counts: dict[str, int] = {}
    for f in files:
        log(f"Đếm nhãn: {f.name}")
        cnt_file = Counter()
        for chunk in pd.read_csv(f, encoding="utf-8", chunksize=1_000_000,
                                 usecols=lambda c: c.strip().lower() == "label"):
            cnt_file.update(_label_column(chunk).value_counts().to_dict())
        label_counts.update(cnt_file)
        row_counts[f.name] = sum(cnt_file.values())

    grouped = Counter()
    for k, v in label_counts.items():
        grouped[LABEL_MAP.get(k, k)] += v

    # ------------------------------------------------------------------ pass 2: mẫu feature
    log(f"Lấy mẫu tối đa {sample_per_file:,} dòng/file để phân tích feature ...")
    samples = []
    for f in files:
        df = pd.read_csv(f, encoding="utf-8", nrows=sample_per_file, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        samples.append(df)
    df = pd.concat(samples, ignore_index=True)
    if "Label" in df.columns:
        df["Label"] = df["Label"].astype(str).str.strip()
    n_sample = len(df)

    nan_cells = int(df.isna().sum().sum())
    dup_rows = int(df.duplicated().sum())
    feature_cols = [c for c in df.columns if c != "Label"]
    df_feat = df[feature_cols].apply(pd.to_numeric, errors="coerce") \
        .replace([np.inf, -np.inf], np.nan)
    nan_feature_rows = int(df_feat.isna().any(axis=1).sum())

    numeric_summary: dict[str, dict] = {}
    present_keys = [c for c in KEY_FEATURES if c in df_feat.columns]
    for c in present_keys:
        s = df_feat[c].replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            continue
        numeric_summary[c] = {
            "mean": float(s.mean()), "std": float(s.std()),
            "min": float(s.min()), "median": float(s.median()), "max": float(s.max()),
        }

    # ------------------------------------------------------------------ biểu đồ
    # 1) phân bố nhãn gốc (top 15)
    top_labels = dict(label_counts.most_common(15))
    fig, ax = plt.subplots(figsize=(8.6, 4.6), constrained_layout=True)
    ax.bar(range(len(top_labels)), list(top_labels.values()), color="#3b76af")
    ax.set_xticks(range(len(top_labels)))
    ax.set_xticklabels(list(top_labels), rotation=35, ha="right", fontsize=8)
    ax.set_yscale("log")
    ax.set_title("Phan bo 15 nhan goc (thang log)")
    for i, v in enumerate(top_labels.values()):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=7, rotation=90)
    fig.savefig(reports_dir / "eda_class_distribution.png", dpi=130)
    plt.close(fig)

    # 2) phân bố 5 lớp chiến thuật
    vals = [grouped.get(g, 0) for g in GROUP_ORDER if g in grouped]
    names = [g for g in GROUP_ORDER if g in grouped]
    fig, ax = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
    colors = ["#57a55a" if n == "BENIGN" else "#d9534f" for n in names]
    ax.bar(names, vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
    ax.set_title("Phan bo 5 lop chien thuat (sau khi gom nhan)")
    fig.savefig(reports_dir / "eda_grouped_distribution.png", dpi=130)
    plt.close(fig)

    # 3) benign/attack theo từng file
    fig, ax = plt.subplots(figsize=(8.6, 4.4), constrained_layout=True)
    file_names = [n.replace(".pcap_ISCX.csv", "").replace("-WorkingHours", "") for n in row_counts]
    benign_part, attack_part = [], []
    for f, n_total in row_counts.items():
        # đếm benign riêng cho file này
        cnt_b = 0
        for chunk in pd.read_csv(data_dir / f, encoding="utf-8", chunksize=1_000_000,
                                 usecols=lambda c: c.strip().lower() == "label"):
            cnt_b += int((_label_column(chunk) == "BENIGN").sum())
        benign_part.append(cnt_b)
        attack_part.append(n_total - cnt_b)
    x = np.arange(len(file_names))
    ax.bar(x, benign_part, label="BENIGN", color="#57a55a")
    ax.bar(x, attack_part, bottom=benign_part, label="Attack", color="#d9534f")
    ax.set_xticks(x)
    ax.set_xticklabels(file_names, rotation=30, ha="right", fontsize=8)
    ax.set_yscale("log")
    ax.legend()
    ax.set_title("Flow benign/attack theo tung ngay (thang log)")
    fig.savefig(reports_dir / "eda_flows_per_file.png", dpi=130)
    plt.close(fig)

    # 4) histogram feature quan trọng
    n_hist = len(present_keys)
    if n_hist:
        cols = 3
        rows = int(np.ceil(n_hist / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(12.5, 3.1 * rows), constrained_layout=True)
        axes = np.atleast_1d(axes).ravel()
        for j, c in enumerate(present_keys):
            s = df_feat[c].replace([np.inf, -np.inf], np.nan).dropna()
            lo, hi = np.percentile(s, [1, 99]) if len(s) > 100 else (s.min(), s.max())
            s_c = s[(s >= lo) & (s <= hi)]
            axes[j].hist(np.log10(s_c[s_c > 1] + 1) if (s_c > 0).all() else s_c, bins=60,
                         color="#3b76af", alpha=0.85)
            axes[j].set_title(c, fontsize=9)
            axes[j].tick_params(labelsize=7)
        for j in range(n_hist, len(axes)):
            axes[j].axis("off")
        fig.suptitle("Phan bo feature quan trong (cat phan Mo 1-99%)", fontsize=11)
        fig.savefig(reports_dir / "eda_feature_distributions.png", dpi=120)
        plt.close(fig)

    # 5) tương quan
    corr_keys = present_keys[:10]
    if len(corr_keys) >= 3:
        corr = df_feat[corr_keys].corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(7.6, 6.4), constrained_layout=True)
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr_keys)))
        ax.set_yticks(range(len(corr_keys)))
        short = [k[:22] for k in corr_keys]
        ax.set_xticklabels(short, rotation=35, ha="right", fontsize=8)
        ax.set_yticklabels(short, fontsize=8)
        for i in range(len(corr_keys)):
            for j in range(len(corr_keys)):
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6)
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title("Ma tran tuong quan (mo dau)")
        fig.savefig(reports_dir / "eda_correlation.png", dpi=130)
        plt.close(fig)

    # ------------------------------------------------------------------ json cho GUI
    info = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "files": [
            {"name": n, "rows": c,
             "counts": {k: int(v) for k, v in
                        Counter({}).items()} if False else None}
            for n, c in row_counts.items()
        ],
        "total_rows": int(sum(row_counts.values())),
        "row_counts": {k: int(v) for k, v in row_counts.items()},
        "label_counts": {k: int(v) for k, v in label_counts.most_common()},
        "grouped_counts": {k: int(v) for k, v in grouped.items()},
        "sample_rows": n_sample,
        "nan_cells_in_sample": nan_cells,
        "nan_feature_rows_in_sample": nan_feature_rows,
        "duplicate_rows_in_sample": dup_rows,
        "numeric_summary": numeric_summary,
        "imbalance_ratio": (max(grouped.values()) / min(grouped.values())) if grouped else None,
    }
    (reports_dir / "eda.json").write_text(json.dumps(info, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    log("Hoàn tất EDA -> " + str(reports_dir))
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--reports-dir", default=str(ROOT / "reports"))
    ap.add_argument("--sample", type=int, default=200_000, help="Số dòng lấy mẫu feature mỗi file")
    args = ap.parse_args()
    info = generate_eda(Path(args.data_dir), Path(args.reports_dir), args.sample)
    print(json.dumps({k: v for k, v in info.items() if k != "numeric_summary"},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
