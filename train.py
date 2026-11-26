#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py — Làm sạch + cân bằng CIC-IDS2017 rồi huấn luyện 6 model:
  Random Forest, Decision Tree, KNN, Linear Logistic Regression, XGBoost, TabR-lite (deep learning).

Kết quả ghi vào ./models/ (mô hình, scaler, danh sách feature) và ./reports/
(classification report, confusion matrix PNG, bảng so sánh model).

Cách chạy:
    python3 train.py                       # tự tải dataset nếu thiếu, train đầy đủ
    python3 train.py --cap 10000           # mỗi lớp tối đa 10.000 mẫu (train nhanh hơn)
    python3 train.py --quick               # chạy thử nhanh (cap 2000, ít epoch)
    python3 train.py --skip-tabr           # bỏ qua mô hình học sâu
    python3 train.py --models rf,xgb       # chỉ train model chỉ định
                                           # (rf, dt, knn, logreg, xgb, tabr)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Cấu hình nhãn: gộp 15 nhãn gốc CIC-IDS2017 thành các lớp chiến thuật
LABEL_MAP = {
    "BENIGN": "BENIGN",
    "DDoS": "DDoS",
    "PortScan": "PortScan",
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "FTP-Patator": "BruteForce",
    "SSH-Patator": "BruteForce",
    # Các lớp dưới đây mặc định KHÔNG dùng (quá nhỏ / khó mô phỏng trong lab):
    # "Web Attack  Brute Force" -> WebAttack, "Web Attack  XSS" -> WebAttack,
    # "Web Attack  Sql Injection" -> WebAttack, "Bot", "Heartbleed", "Infiltration"
}
DEFAULT_CLASSES = ["BENIGN", "DoS", "DDoS", "PortScan", "BruteForce"]
ALL_MODELS = ["rf", "dt", "knn", "logreg", "xgb", "tabr"]
# ---------------------------------------------------------------------------


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_dataset(data_dir: Path, classes: list[str], cap: int, benign_cap: int = 0) -> pd.DataFrame:
    """Đọc từng file theo chunk, lọc lớp + cap ngay khi đọc (an toàn bộ nhớ).

    Trả về DataFrame đã làm sạch (numeric, bỏ NaN/inf, bỏ trùng lặp).
    Nếu cap <= 0 thì giữ toàn bộ các lớp tấn công; benign_cap > 0 giới hạn riêng lớp BENIGN.
    """
    files = sorted(data_dir.glob("*.csv"))
    if not files:
        log(f"Không có file CSV trong {data_dir}. Chạy: python3 download_dataset.py")
        sys.exit(1)
    counts = {c: 0 for c in classes}
    parts: list[pd.DataFrame] = []
    for f in files:
        log(f"Đọc {f.name} ...")
        for chunk in pd.read_csv(f, encoding="utf-8", chunksize=300_000, low_memory=False):
            chunk.columns = [str(c).strip() for c in chunk.columns]
            if "Label" not in chunk.columns:
                log("Thiếu cột 'Label' — dataset không đúng định dạng CIC-IDS2017?")
                sys.exit(1)
            # bỏ dòng header lặp lại
            chunk = chunk[chunk["Label"].astype(str).str.strip().ne("Label")]
            chunk["Label"] = chunk["Label"].astype(str).str.strip()
            chunk["Label"] = chunk["Label"].map(LABEL_MAP).fillna(chunk["Label"])
            chunk = chunk[chunk["Label"].isin(classes)]
            if chunk.empty:
                continue
            limits = {
                c: (benign_cap if (c == "BENIGN" and benign_cap > 0) else (cap if cap > 0 else 10**12))
                for c in classes
            }
            keep = []
            for cls, g in chunk.groupby("Label", sort=False):
                remaining = limits[cls] - counts[cls]
                if remaining > 0:
                    keep.append(g.head(remaining))
                    counts[cls] += min(len(g), remaining)
            if not keep:
                continue
            kc = pd.concat(keep)
            feature_cols = [c for c in kc.columns if c != "Label"]
            kc[feature_cols] = kc[feature_cols].apply(pd.to_numeric, errors="coerce")
            kc = kc.replace([np.inf, -np.inf], np.nan).dropna().drop_duplicates()
            if not kc.empty:
                kc[feature_cols] = kc[feature_cols].astype(np.float32)
                parts.append(kc)
    df = pd.concat(parts, ignore_index=True)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    log(f"Sau khi đọc + làm sạch (cap={'ALL' if cap <= 0 else cap}, benign_cap={benign_cap or 'ALL'}): {len(df):,} dòng")
    for cls, c in counts.items():
        log(f"    {cls:<12} {c:,}")
    return df


def build_models(args, n_classes: int) -> dict:
    models: dict = {}
    if "rf" in args.models:
        models["rf"] = RandomForestClassifier(
            n_estimators=300, max_depth=None, n_jobs=-1, random_state=42, class_weight="balanced_subsample")
    if "dt" in args.models:
        models["dt"] = DecisionTreeClassifier(
            max_depth=None, min_samples_leaf=2, random_state=42, class_weight="balanced")
    if "knn" in args.models:
        models["knn"] = KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1)
    if "logreg" in args.models:
        models["logreg"] = LogisticRegression(
            max_iter=3000, C=1.0, random_state=42, n_jobs=-1)
    if "xgb" in args.models:
        try:
            from xgboost import XGBClassifier
        except ImportError:
            log("Thiếu xgboost: pip install xgboost — bỏ qua model xgb")
            return models
        models["xgb"] = XGBClassifier(
            n_estimators=400, max_depth=8, learning_rate=0.15, subsample=0.9,
            colsample_bytree=0.9, tree_method="hist", n_jobs=-1,
            random_state=42, eval_metric="mlogloss")
    return models


def evaluate_model(name: str, model, X_test, y_test, le, reports_dir: Path,
                   features: list[str]) -> dict:
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    mf1 = f1_score(y_test, pred, average="macro")
    wf1 = f1_score(y_test, pred, average="weighted")
    rep = classification_report(y_test, pred, target_names=list(le.classes_), digits=4)
    (reports_dir / f"report_{name}.txt").write_text(
        f"Model: {name}\nAccuracy: {acc:.4f} | Macro-F1: {mf1:.4f} | Weighted-F1: {wf1:.4f}\n\n"
        + rep, encoding="utf-8")
    cm = confusion_matrix(y_test, pred)
    plot_confusion(name, cm, list(le.classes_), reports_dir)
    log(f"    {name:<8} acc={acc:.4f}  macroF1={mf1:.4f}  weightedF1={wf1:.4f}")
    return {"accuracy": acc, "macro_f1": mf1, "weighted_f1": wf1,
            "confusion_matrix": cm.tolist(),
            "labels": list(le.classes_),
            "feature_names": features}


def plot_confusion(name: str, cm, labels, reports_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 5.4), constrained_layout=True)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_title(f"Confusion Matrix - {name}", fontsize=11)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.savefig(reports_dir / f"confusion_{name}.png", dpi=130)
    plt.close(fig)


def plot_comparison(metrics: dict, reports_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(metrics)
    accs = [metrics[n]["accuracy"] for n in names]
    mf1s = [metrics[n]["macro_f1"] for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    ax.bar(x - 0.2, accs, 0.4, label="Accuracy", color="#3b76af")
    ax.bar(x + 0.2, mf1s, 0.4, label="Macro-F1", color="#e07b39")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.set_title("Model comparison on CIC-IDS2017 (balanced 5-class)")
    for xi, (a, m) in enumerate(zip(accs, mf1s)):
        ax.text(xi - 0.2, a + 0.01, f"{a:.3f}", ha="center", fontsize=8)
        ax.text(xi + 0.2, m + 0.01, f"{m:.3f}", ha="center", fontsize=8)
    fig.savefig(reports_dir / "model_comparison.png", dpi=130)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Train IDS models trên CIC-IDS2017")
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--out-dir", default=str(ROOT / "models"))
    ap.add_argument("--reports-dir", default=str(ROOT / "reports"))
    ap.add_argument("--cap", type=int, default=20000, help="Số mẫu tối đa mỗi lớp (0 = lấy toàn bộ)")
    ap.add_argument("--benign-cap", type=int, default=0, help="Giới hạn riêng số mẫu lớp BENIGN (0 = theo --cap)")
    ap.add_argument("--test-size", type=float, default=0.20, help="Tỷ lệ tập test (mặc định 0.20 = 20%%)")
    ap.add_argument("--quick", action="store_true", help="Chạy nhanh để thử pipeline")
    ap.add_argument("--skip-tabr", action="store_true")
    ap.add_argument("--models", default=",".join(ALL_MODELS),
                    help="Danh sách model: rf,dt,knn,logreg,xgb,tabr")
    ap.add_argument("--tabr-epochs", type=int, default=12)
    ap.add_argument("--skip-eda", action="store_true", help="Không sinh báo cáo EDA sau khi train")
    args = ap.parse_args()

    if args.quick:
        args.cap = min(args.cap, 2000)
        args.tabr_epochs = min(args.tabr_epochs, 2)

    models_wanted = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in models_wanted:
        if m not in ALL_MODELS:
            log(f"Model '{m}' không hợp lệ. Chọn trong: {ALL_MODELS}")
            return 1
    if args.skip_tabr:
        models_wanted = [m for m in models_wanted if m != "tabr"]

    data_dir, out_dir, reports_dir = Path(args.data_dir), Path(args.out_dir), Path(args.reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 0. Tự tải dataset nếu thiếu
    if not any(data_dir.glob("*.csv")):
        log("Thư mục data/ trống — gọi download_dataset.py ...")
        import download_dataset as dl
        if dl.main() != 0:
            return 1

    # 1. Nạp + làm sạch + cân bằng (stream: lọc và cap ngay khi đọc)
    t0 = time.time()
    df = load_dataset(data_dir, DEFAULT_CLASSES, args.cap, benign_cap=args.benign_cap)
    if df.empty:
        log("Không còn dữ liệu sau khi lọc — kiểm tra các file CSV trong data/")
        return 1

    # 2. Tách X/y, scale, chia train/test stratified (mặc định 80% train / 20% test)
    import gc
    features = [c for c in df.columns if c != "Label"]
    X = df[features].to_numpy(dtype=np.float32)
    le = LabelEncoder()
    y = le.fit_transform(df["Label"].to_numpy())
    del df
    gc.collect()
    log(f"Ma trận dữ liệu: {X.shape}, {len(le.classes_)} lớp")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y)
    del X
    gc.collect()

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train).astype(np.float32, copy=False)
    X_test_s = scaler.transform(X_test).astype(np.float32, copy=False)
    del X_train, X_test
    gc.collect()
    log(f"Chia dữ liệu ({100*(1-args.test_size):.0f}/{100*args.test_size:.0f}): "
        f"train={len(X_train_s):,} test={len(X_test_s):,}")

    # 3. Train từng model
    metrics: dict = {}
    eval_ctx = (le, reports_dir, features)
    torch_ok = True
    for name in models_wanted:
        log(f"== Train {name} ==")
        try:
            if name == "tabr":
                if not torch_ok:
                    log("    PyTorch chưa cài — bỏ qua tabr")
                    continue
                try:
                    from tabr_model import train_tabr
                except ImportError as e:
                    log(f"    Không nạp được PyTorch ({e}) — bỏ qua tabr")
                    torch_ok = False
                    continue
                # Tách 10% từ tập train làm validation cho early stopping (giữ nguyên 20% test)
                X_tr_tabr, X_val_tabr, y_tr_tabr, y_val_tabr = train_test_split(
                    X_train_s, y_train, test_size=0.10, random_state=42, stratify=y_train)
                tabr, _ = train_tabr(
                    X_tr_tabr, y_tr_tabr, X_val_tabr, y_val_tabr,
                    n_classes=len(le.classes_),
                    epochs=args.tabr_epochs, verbose=True)
                pred = tabr.predict_proba(X_test_s).argmax(axis=1) \
                    if hasattr(tabr, "predict_proba") else None
                # evaluate thông qua wrapper dự đoán trực tiếp bằng torch
                import torch as _torch
                with _torch.no_grad():
                    logits = []
                    for i in range(0, len(X_test_s), 4096):
                        xb = _torch.tensor(X_test_s[i:i + 4096], dtype=_torch.float32)
                        logits.append(tabr(xb).numpy())
                pred = np.concatenate(logits).argmax(axis=1)
                acc = accuracy_score(y_test, pred)
                mf1 = f1_score(y_test, pred, average="macro")
                wf1 = f1_score(y_test, pred, average="weighted")
                rep = classification_report(y_test, pred, target_names=list(le.classes_), digits=4)
                (reports_dir / "report_tabr.txt").write_text(
                    f"Model: tabr (TabR-lite)\nAccuracy: {acc:.4f} | Macro-F1: {mf1:.4f} "
                    f"| Weighted-F1: {wf1:.4f}\n\n" + rep, encoding="utf-8")
                cm = confusion_matrix(y_test, pred)
                plot_confusion("tabr", cm, list(le.classes_), reports_dir)
                log(f"    tabr     acc={acc:.4f}  macroF1={mf1:.4f}  weightedF1={wf1:.4f}")
                metrics["tabr"] = {"accuracy": acc, "macro_f1": mf1, "weighted_f1": wf1,
                                   "confusion_matrix": cm.tolist(),
                                   "labels": list(le.classes_),
                                   "feature_names": features}
                tabr.save(out_dir / "tabr.pt", {
                    "n_features": X_train_s.shape[1], "n_classes": len(le.classes_),
                    "d_token": 64, "n_blocks": 2, "n_heads": 4,
                    "knn_alpha": float(tabr.knn_alpha)})
                joblib.dump(le, out_dir / "label_encoder_tabr.joblib")
            else:
                sk_models = build_models(args, len(le.classes_))
                if name not in sk_models:
                    log(f"    Không dựng được model '{name}' (thiếu thư viện?)")
                    continue
                model = sk_models[name]
                t1 = time.time()
                model.fit(X_train_s, y_train)
                log(f"    fit xong sau {time.time()-t1:.1f}s")
                metrics[name] = evaluate_model(name, model, X_test_s, y_test, *eval_ctx)
                joblib.dump(model, out_dir / f"{name}.joblib")
        except Exception as e:  # noqa: BLE001
            log(f"    [!] Lỗi khi train {name}: {e}")
            import traceback
            traceback.print_exc()

    # 4. Lưu artifact dùng chung + báo cáo tổng hợp
    joblib.dump(scaler, out_dir / "scaler.joblib")
    joblib.dump(le, out_dir / "label_encoder.joblib")
    (out_dir / "features.json").write_text(json.dumps(features, ensure_ascii=False, indent=1),
                                           encoding="utf-8")
    (out_dir / "classes.json").write_text(json.dumps(list(le.classes_), ensure_ascii=False),
                                          encoding="utf-8")
    if metrics:
        # Gộp với metrics từ các lần train trước (train theo đợt vẫn giữ đủ báo cáo)
        mpath = reports_dir / "metrics.json"
        combined = {}
        if mpath.exists():
            try:
                old = json.loads(mpath.read_text())
                # chỉ giữ model cũ nếu nó được train cùng cấu hình (cùng số lớp)
                same_setup = all(v.get("labels") == list(le.classes_) for v in old.values())
                if same_setup:
                    combined.update(old)
            except Exception:  # noqa: BLE001
                pass
        combined.update({k: {kk: vv for kk, vv in v.items()} for k, v in metrics.items()})
        (reports_dir / "metrics.json").write_text(
            json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "confusion_matrix"}
                        for k, v in combined.items()}, indent=1), encoding="utf-8")
        pd.DataFrame({n: {kk: round(vv, 4) for kk, vv in v.items()
                          if kk in ("accuracy", "macro_f1", "weighted_f1")}
                      for n, v in combined.items()}).T.to_csv(reports_dir / "metrics.csv")
        plot_comparison(combined, reports_dir)
        log("So sánh model (test set):")
        for n, m in combined.items():
            log(f"    {n:<8} acc={m['accuracy']:.4f}  macroF1={m['macro_f1']:.4f}")

    # 5. Sinh luôn báo cáo EDA cho tab EDA trong GUI (bỏ qua nếu --skip-eda)
    if not args.skip_eda:
        try:
            from eda import generate_eda
            log("Sinh báo cáo EDA cho GUI ...")
            generate_eda(data_dir, reports_dir, sample_per_file=150_000)
        except Exception as e:  # noqa: BLE001
            log(f"[!] Không sinh được EDA (không nghiêm trọng): {e}")

    log(f"Hoàn tất sau {time.time()-t0:.0f}s. Model ở {out_dir}, báo cáo ở {reports_dir}")
    log("Mở giao diện:   python3 nids_gui.py")
    log("Real-time CLI:  python3 realtime_detector.py --replay <file.csv> (thử nhanh)")
    log("                python3 realtime_detector.py --dir ~/nids/flows  (CICFlowMeter live)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
