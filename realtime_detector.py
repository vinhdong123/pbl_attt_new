#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
realtime_detector.py — Bộ phát hiện tấn công REAL-TIME trên Kali.

Nguồn dữ liệu: các file CSV mà CICFlowMeter ghi liên tục khi bắt gói trên interface
(host-only VMnet1). Script "đuôi theo" (tail) các file này, chuyển từng flow thành
vector 78 feature giống hệt lúc train, cho chạy qua toàn bộ model đã huấn luyện
(RandomForest, DecisionTree, KNN, LogisticRegression, XGBoost, TabR-lite),
gộp kết quả (ensemble trung bình xác suất) và IN CẢNH BÁO RA MÀN HÌNH.

Cách chạy:
    # 1) Thử nhanh mà chưa cần CICFlowMeter — "phát lại" 1 file CSV bất kỳ:
    python3 realtime_detector.py --replay data/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv

    # 2) Chạy live với CICFlowMeter (đặt Output Directory của CICFlowMeter = ~/nids/flows):
    python3 realtime_detector.py --dir ~/nids/flows

    # Tuỳ chọn hữu ích:
    #   --threshold 0.5     ngưỡng tin cậy để报警 (ensemble mean prob)
    #   --show-benign       in cả flow bình thường (mặc định chỉ đếm)
    #   --no-color          tắt màu ANSI
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, deque
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent

COLORS = {
    "BENIGN": "\033[90m",
    "PortScan": "\033[93m",
    "DoS": "\033[91m",
    "DDoS": "\033[91;1m",
    "BruteForce": "\033[95m",
    "ATTACK": "\033[91;7m",
}
RESET = "\033[0m"
ATTACK_EMOJI = {"PortScan": "SCAN", "DoS": "DOS", "DDoS": "DDOS", "BruteForce": "BRUTE"}


def c(text: str, key: str, use_color: bool) -> str:
    return f"{COLORS.get(key, '')}{text}{RESET}" if use_color else text


# ---------------------------------------------------------------------------
# Nạp model + scaler + metadata
# ---------------------------------------------------------------------------
class ModelBank:
    def __init__(self, models_dir: Path, use_color: bool = True):
        self.use_color = use_color
        if not (models_dir / "features.json").exists():
            print("Không tìm thấy model trong", models_dir)
            print(">>> Chạy 'python3 train.py' trước (xem README.md).")
            sys.exit(1)
        self.features: list[str] = json.loads((models_dir / "features.json").read_text())
        self.classes: list[str] = json.loads((models_dir / "classes.json").read_text())
        self.scaler = joblib.load(models_dir / "scaler.joblib")
        self.models: dict[str, object] = {}
        for p in sorted(models_dir.glob("*.joblib")):
            if p.stem.startswith(("scaler", "label_encoder")):
                continue  # các file tiện ích, không phải model dự đoán
            self.models[p.stem] = joblib.load(p)
        # TabR (deep learning) — nếu PyTorch có mặt
        self.tabr = None
        if (models_dir / "tabr.pt").exists():
            try:
                from tabr_model import TabRLite
                self.tabr = TabRLite.load(models_dir / "tabr.pt")
            except ImportError as e:
                print(f"[!] PyTorch chưa cài, bỏ qua TabR ({e})")
        names = ", ".join(self.models) + (" + tabr" if self.tabr else "")
        print(f"[✓] Nạp {len(self.models) + (1 if self.tabr else 0)} model: {names}")
        print(f"[✓] {len(self.features)} feature | các lớp: {self.classes}")

    def _row_to_vec(self, row: dict, warned: set) -> np.ndarray | None:
        lut = {k.strip().lower(): v for k, v in row.items()}
        vals, miss = [], 0
        for f in self.features:
            v = lut.get(f.lower())
            if v is None or v == "":
                miss += 1
                vals.append(0.0)
                continue
            try:
                vals.append(float(v))
            except ValueError:
                vals.append(0.0)
        if miss and ("miss" not in warned):
            warned.add("miss")
            print(f"[~] {miss}/{len(self.features)} feature không có trong CSV đầu vào "
                  f"— điền 0 (kiểm tra cài đặt CICFlowMeter)")
        vec = np.array([0.0 if (np.isnan(x) or np.isinf(x)) else x for x in vals],
                       dtype=np.float64)
        return vec

    def predict(self, vec: np.ndarray) -> tuple[str, float, dict]:
        """Dự đoán 1 flow (tiện cho CLI). Xem predict_batch cho bản theo lô."""
        labels, confs, _, votes = self.predict_batch(vec.reshape(1, -1))
        detail = {name: (vote_labels[0], float(vote_confs[0]))
                  for name, (vote_labels, vote_confs) in votes.items()}
        return labels[0], float(confs[0]), detail

    def predict_batch(self, vecs: np.ndarray) -> tuple[list[str], np.ndarray, np.ndarray, dict]:
        """Dự đoán N flow cùng lúc (nhanh hơn nhiều lần so với từng flow).

        Trả về: (nhãn, độ tin cậy, ma trận xác suất, chi tiết từng model
                 dạng {tên_model: (danh_sách_nhãn, mảng_conf)}).
        """
        v = self.scaler.transform(vecs)
        n = len(v)
        proba_sum = np.zeros((n, len(self.classes)), dtype=np.float64)
        votes: dict = {}
        used = 0
        for name, model in self.models.items():
            try:
                p = model.predict_proba(v)
                if p.shape[1] != len(self.classes):  # model thiếu lớp
                    full = np.zeros((n, len(self.classes)))
                    full[:, :p.shape[1]] = p
                    p = full
            except Exception:
                continue
            proba_sum += p
            used += 1
            idx = p.argmax(axis=1)
            votes[name] = ([self.classes[int(i)] for i in idx], p.max(axis=1))
        if self.tabr is not None:
            try:
                import torch
                logits = []
                with torch.no_grad():
                    for i in range(0, n, 256):
                        xt = torch.tensor(v[i:i + 256], dtype=torch.float32)
                        logits.append(self.tabr(xt).numpy())
                logits = np.concatenate(logits)
                p = np.exp(logits - logits.max(axis=1, keepdims=True))
                p = p / p.sum(axis=1, keepdims=True)
                proba_sum += p
                used += 1
                idx = p.argmax(axis=1)
                votes["tabr"] = ([self.classes[int(i)] for i in idx], p.max(axis=1))
            except Exception:
                pass
        proba = proba_sum / max(used, 1)
        idx = proba.argmax(axis=1)
        conf = proba.max(axis=1)
        labels = [self.classes[int(i)] for i in idx]
        return labels, conf, proba, votes


# ---------------------------------------------------------------------------
# Tail file CSV đang lớn dần (CICFlowMeter ghi liên tục)
# ---------------------------------------------------------------------------
class FileTailer:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.offset = 0
        self.header: list[str] | None = None
        self.buf = b""

    def poll(self, max_rows: int = 2000) -> list[dict]:
        try:
            size = self.path.stat().st_size
        except OSError:
            return []
        if size < self.offset:  # file bị xoá/ghi lại từ đầu
            self.offset, self.header, self.buf = 0, None, b""
        with open(self.path, "rb") as f:
            f.seek(self.offset)
            data = f.read()
        if not data:
            return []
        self.offset += len(data)
        self.buf += data
        lines = self.buf.split(b"\n")
        self.buf = lines.pop()
        rows: list[dict] = []
        for ln in lines:
            ln = ln.strip(b"\r").decode("utf-8", errors="replace")
            if not ln:
                continue
            if self.header is None:
                self.header = [h.strip() for h in ln.split(",")]
                continue
            vals = ln.split(",")
            if len(vals) != len(self.header):
                vals = (vals + [""] * len(self.header))[:len(self.header)]
            rows.append(dict(zip(self.header, vals)))
            if len(rows) >= max_rows:
                break
        return rows


def watch_dir(dir_path: Path):
    """Sinh ra (path, rows) cho mọi file CSV mới/cập nhật trong thư mục."""
    tails: dict[Path, FileTailer] = {}
    known = set()
    while True:
        try:
            current = {p for p in dir_path.glob("*.csv") if p.stat().st_size > 0}
        except OSError:
            current = set()
        for new in current - known:
            print(f"[watch] Theo dõi file mới: {new.name}")
            tails[new] = FileTailer(new)
        known = current
        for path, t in list(tails.items()):
            if path not in known:
                del tails[path]
                continue
            rows = t.poll()
            if rows:
                yield path, rows
        time.sleep(1.0)


# ---------------------------------------------------------------------------
# Hiển thị cảnh báo
# ---------------------------------------------------------------------------
class AlertDisplay:
    def __init__(self, use_color: bool, window: int, show_benign: bool):
        self.use_color = use_color
        self.window = window
        self.show_benign = show_benign
        self.recent: deque = deque()          # (time, label, src)
        self.n_benign = 0
        self.n_attack = 0
        self.n_attack_by_class: Counter = Counter()

    def push(self, label: str, conf: float, detail: dict, row: dict, row_no: int):
        now = time.time()
        src = row.get("Source IP") or row.get("SourceIP") or row.get("Src IP") or "?"
        dst_ip = row.get("Destination IP") or row.get("DestIP") or row.get("Dst IP") or ""
        dst_port = row.get("Destination Port") or row.get("Dst Port") or ""
        ts = row.get("Timestamp") or row.get("Flow ID") or f"row#{row_no}"

        if label == "BENIGN":
            self.n_benign += 1
            self.recent.append((now, "BENIGN", str(src)))
            if self.show_benign:
                print(c(f"    benign  conf={conf:.2f}  {ts}  {src}", "BENIGN", self.use_color))
            return

        self.n_attack += 1
        self.n_attack_by_class[label] += 1
        self.recent.append((now, label, str(src)))
        votes = " ".join(f"{k}:{v[0][:4]}({v[1]:.2f})" for k, v in detail.items())
        dst = f"{dst_ip}:{dst_port}" if dst_ip else f"port {dst_port}"
        head = ATTACK_EMOJI.get(label, "ATTACK")
        line = (f"{time.strftime('%H:%M:%S')} [{head}] {label}  "
                f"conf={conf:.2f}  {src} -> {dst}")
        print(c(">>> " + line, label if label in COLORS else "ATTACK", self.use_color))
        print(c(f"      chi tiết model: {votes}", "BENIGN", self.use_color))

    def maybe_summary(self, force=False):
        cutoff = time.time() - self.window
        while self.recent and self.recent[0][0] < cutoff:
            self.recent.popleft()
        counts = Counter(lbl for _, lbl, _ in self.recent)
        if force or counts.get("BENIGN", 0) + sum(v for k, v in counts.items() if k != "BENIGN"):
            attacks = {k: v for k, v in counts.items() if k != "BENIGN"}
            top_src = Counter(src for _, lbl, src in self.recent if lbl != "BENIGN").most_common(1)
            msg = (f"[tổng] benign={self.n_benign:,}  attack={self.n_attack:,}  "
                   f"({dict(self.n_attack_by_class)})")
            if attacks:
                msg += f" | {self.window}s qua: {dict(attacks)}"
                if top_src:
                    msg += f" | nguồn đáng ngờ: {top_src[0][0]} ({top_src[0][1]} flow)"
            print(c(msg, "BENIGN", self.use_color))


# ---------------------------------------------------------------------------
def run_stream(bank: ModelBank, display: AlertDisplay, rows_iter, threshold: float,
               max_rate: float | None = None, min_gap: float = 0.0):
    warned: set = set()
    n = 0
    t_last = time.time()
    for row_no, row in rows_iter:
        vec = bank._row_to_vec(row, warned)
        if vec is None:
            continue
        label, conf, detail = bank.predict(vec)
        display.push(label, conf, detail, row, row_no)
        n += 1
        if max_rate:  # phát lại có độ trễ để quan sát màn hình
            target = t_last + 1.0 / max_rate
            dt = target - time.time()
            if dt > 0:
                time.sleep(dt)
            t_last = time.time()
        if min_gap and time.time() - t_last < min_gap:
            time.sleep(min_gap)
    display.maybe_summary(force=True)
    print(f"\nXử lý xong {n:,} flow.")


def replay_iter(path: Path, chunk: int = 500):
    """Đọc file CSV tĩnh theo từng khúc — dùng để test không cần CICFlowMeter."""
    import pandas as pd
    df_iter = pd.read_csv(path, encoding="utf-8", engine="c", chunksize=chunk, low_memory=False)
    idx = 0
    for chunk_df in df_iter:
        chunk_df.columns = [str(c).strip() for c in chunk_df.columns]
        for _, r in chunk_df.iterrows():
            idx += 1
            yield idx, {k: ("" if pd.isna(v) else str(v)) for k, v in r.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description="Real-time IDS detector (Kali)")
    ap.add_argument("--iface", default=None, help="Bắt gói tin trực tiếp trên card mạng (vd: eth0, ens33, any)")
    ap.add_argument("--dir", default="~/nids/flows", help="Thư mục CICFlowMeter ghi CSV live")
    ap.add_argument("--models-dir", default=str(ROOT / "models"))
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--window", type=int, default=60, help="Cửa sổ tóm tắt (giây)")
    ap.add_argument("--no-color", action="store_true")
    ap.add_argument("--show-benign", action="store_true", help="In cả flow benign")
    ap.add_argument("--replay", default=None, help="File CSV để phát lại (test không cần CICFlowMeter)")
    ap.add_argument("--replay-speed", type=float, default=0, help="Số flow/giây khi phát lại (0 = nhanh nhất)")
    args = ap.parse_args()

    use_color = not args.no_color and sys.stdout.isatty()
    bank = ModelBank(Path(args.models_dir), use_color)
    display = AlertDisplay(use_color, args.window, args.show_benign)

    if args.iface:
        import threading
        from live_sniffer import LiveFlowSniffer, ensure_lab_services
        svcs = ensure_lab_services()
        if svcs:
            print(f"[✓] Dịch vụ lab sẵn sàng: {', '.join(svcs)}")
        stop_ev = threading.Event()
        sniffer = LiveFlowSniffer(iface=args.iface)
        print(f"[live-sniffer] Đang bắt gói trực tiếp trên '{args.iface}' — Ctrl+C để dừng. Threshold={args.threshold}")
        try:
            def sniffer_iter():
                row_no = 0
                last_summary = time.time()
                for rows in sniffer.stream_flows(stop_ev):
                    for row in rows:
                        row_no += 1
                        yield row_no, row
                    if time.time() - last_summary >= 10:
                        display.maybe_summary()
                        last_summary = time.time()
            run_stream(bank, display, sniffer_iter(), args.threshold)
        except KeyboardInterrupt:
            stop_ev.set()
            print("\nDừng detector. Tổng kết phiên:")
            display.maybe_summary(force=True)
        except PermissionError as e:
            print(f"[!] {e}")
            return 1
        return 0

    if args.replay:
        path = Path(args.replay).expanduser()
        if not path.exists():
            print("Không thấy file:", path)
            return 1
        print(f"[replay] Phát lại {path.name} (speed={args.replay_speed} flow/s)...")
        run_stream(bank, display, replay_iter(path), args.threshold,
                   max_rate=args.replay_speed or None)
        return 0

    dir_path = Path(args.dir).expanduser()
    if not dir_path.exists():
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"[!] Đã tạo {dir_path} — trống hiện tại.")
        print("    Hãy chạy CICFlowMeter và đặt Output Directory = thư mục này")
        print("    (xem README.md mục 'Chạy real-time'). Đang chờ file CSV ...")
    print(f"[live] Theo dõi {dir_path} — Ctrl+C để dừng. Threshold={args.threshold}")
    try:
        def live_iter():
            row_no = 0
            last_summary = time.time()
            for _path, rows in watch_dir(dir_path):
                for row in rows:
                    row_no += 1
                    yield row_no, row
                if time.time() - last_summary >= 10:
                    display.maybe_summary()
                    last_summary = time.time()
        run_stream(bank, display, live_iter(), args.threshold)
    except KeyboardInterrupt:
        print("\nDừng detector. Tổng kết phiên:")
        display.maybe_summary(force=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
