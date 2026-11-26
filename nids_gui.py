#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nids_gui.py — Giao diện phần mềm NIDS (Kali Linux) gồm 3 tab:
  1. "So sanh mo hinh"   : so sánh độ chính xác 5 model baseline (RF, DT, KNN, LogReg, XGB)
                           với mô hình học sâu TabR-lite.
  2. "EDA du lieu"       : phân tích khám phá dataset CIC-IDS2017 (biểu đồ + thống kê).
  3. "Phat hien real-time": khung hiển thị lưu lượng đến theo thời gian thực + cảnh báo.

Cách chạy trên Kali:
    sudo apt install python3-tk        # nếu chưa có Tkinter
    python3 nids_gui.py                # mở giao diện
    python3 nids_gui.py --smoke        # chế độ tự kiểm tra (tự đóng sau 3 giây)

Ghi chú kỹ thuật: nếu matplotlib-TkAgg không nạp được trong môi trường hiện tại
(vd cài lệch Tcl/Tk), chương trình TỰ ĐỘNG chuyển sang vẽ bằng Tk thuần
(tk.Canvas + tk.PhotoImage) nên GUI vẫn chạy được.
"""
from __future__ import annotations

import argparse
import json
import queue as q
import sys
import threading
import time
from pathlib import Path

import numpy as np

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Matplotlib-TkAgg: dùng nếu nạp được, không thì fallback Tk thuần
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    import matplotlib.image as mpimg
    HAS_MPL_TK = True
except Exception:  # noqa: BLE001
    HAS_MPL_TK = False

from realtime_detector import ModelBank, FileTailer, replay_iter  # noqa: E402
from live_sniffer import LiveFlowSniffer, ensure_lab_services, list_interfaces  # noqa: E402

FONT_UI = "Segoe UI" if sys.platform.startswith("win") else "DejaVu Sans"
FONT_MONO = "Consolas" if sys.platform.startswith("win") else "DejaVu Sans Mono"

BG = "#f8f9fa"
FG = "#111827"
MUTED = "#4b5563"
PANEL = "#ffffff"
GRIDC = "#d1d5db"
ACCENT = "#111827"
ORANGE = "#6b7280"
BAD = "#b91c1c"
OKG = "#15803d"

ATTACK_COLOR = {"PortScan": "#b45309", "DoS": "#b91c1c", "DDoS": "#991b1b",
                "BruteForce": "#6d28d9"}

MODEL_LABELS = {
    "tabr": "TabR-lite (Deep Learning)",
    "xgb": "XGBoost",
    "rf": "Random Forest",
    "dt": "Decision Tree",
    "knn": "K-Nearest Neighbors",
    "logreg": "Linear Logistic Regression",
}
MODEL_SHORT = {
    "tabr": "TabR-lite\n(DL)",
    "xgb": "XGBoost",
    "rf": "Random\nForest",
    "dt": "Decision\nTree",
    "knn": "KNN",
    "logreg": "Linear\nLogReg",
}
LABEL_TO_KEY = {v: k for k, v in MODEL_LABELS.items()}


# ===========================================================================
# Worker dự đoán chạy trong thread nền
# ===========================================================================
class DetectorWorker(threading.Thread):
    """Đọc flow (live hoặc replay), dự đoán bằng ensemble, đẩy sự kiện vào queue."""

    def __init__(self, models_dir: Path, flow_dir: Path, replay_path: Path | None,
                 threshold: float, show_benign: bool, speed: float,
                 event_q: "q.Queue", stop_event: threading.Event, iface: str | None = None):
        super().__init__(daemon=True)
        self.models_dir, self.flow_dir = models_dir, flow_dir
        self.replay_path, self.threshold = replay_path, threshold
        self.show_benign, self.speed = show_benign, speed
        self.event_q, self.stop_event = event_q, stop_event
        self.iface = iface

    def run(self):
        try:
            bank = ModelBank(self.models_dir)
        except SystemExit:
            self.event_q.put(("status", "Chua co model — hay chay train.py truoc"))
            return
        warned: set = set()
        n_benign, n_attack = 0, 0
        pending: list[tuple[int, dict]] = []
        t_last = time.time()
        # Cửa sổ trượt 10s theo IP nguồn để nhận diện mẫu hành vi nhiều flow (BruteForce/PortScan/DoS)
        from collections import defaultdict, deque
        win_ports: dict[str, deque] = defaultdict(deque)      # src -> [(ts, dport)]
        win_auth: dict[tuple[str, int], deque] = defaultdict(deque)  # (src, dport) -> [ts]
        win_http: dict[str, deque] = defaultdict(deque)       # src -> [ts]

        def process_pending(force=False):
            """Dự đoán cả lô flow cùng lúc — nhanh hơn nhiều lần so với từng flow."""
            nonlocal n_benign, n_attack, t_last
            if not pending:
                return
            if not force and len(pending) < 16 and time.time() - t_last < 0.15:
                return
            take, pending[:] = pending[:512], pending[512:]
            t_last = time.time()
            vecs, valid = [], []
            for row_no, row in take:
                vec = bank._row_to_vec(row, warned)
                if vec is not None:
                    vecs.append(vec)
                    valid.append((row_no, row))
            if not valid:
                return
            import numpy as np
            matrix = np.vstack(vecs)
            labels, confs, _proba, votes = bank.predict_batch(matrix)
            items = []
            now_mono = time.time()
            for i, ((row_no, row), label, conf) in enumerate(zip(valid, labels, confs)):
                detail = " ".join(f"{name}:{vote_l[i][:4]}({vote_c[i]:.2f})"
                                  for name, (vote_l, vote_c) in votes.items())
                n_fwd = int(float(row.get("Total Fwd Packets") or 0))
                n_bwd = int(float(row.get("Total Backward Packets") or 0))
                b_fwd = int(float(row.get("Total Length of Fwd Packets") or 0))
                b_bwd = int(float(row.get("Total Length of Bwd Packets") or 0))
                syn_c = int(float(row.get("SYN Flag Count") or 0))
                pps = float(row.get("Flow Packets/s") or 0)
                pkts = n_fwd + n_bwd
                n_bytes = b_fwd + b_bwd
                ts_now = row.get("Timestamp") or row.get("Flow ID") or time.strftime("%H:%M:%S")
                src_ip = row.get("Source IP") or row.get("SourceIP") or row.get("Src IP") or "-"
                dst_ip = row.get("Destination IP") or row.get("DestIP") or row.get("Dst IP") or "-"
                dport = row.get("Destination Port") or row.get("Dst Port") or "-"
                try:
                    dp_int = int(float(dport))
                except (TypeError, ValueError):
                    dp_int = -1

                # Kết hợp hành vi đa-flow khi chạy live (giải quyết lệch RTT VMware)
                conf_val = float(conf)
                if self.iface and src_ip != "-" and dp_int > 0:
                    qp = win_ports[src_ip]
                    qp.append((now_mono, dp_int))
                    while qp and now_mono - qp[0][0] > 10.0:
                        qp.popleft()
                    if dp_int in (21, 22):
                        qa = win_auth[(src_ip, dp_int)]
                        qa.append(now_mono)
                        while qa and now_mono - qa[0] > 10.0:
                            qa.popleft()
                        if len(qa) >= 4:
                            label = "BruteForce"
                            conf_val = max(conf_val, 0.88)
                            detail = f"Hybrid(auth={len(qa)}/10s) " + detail
                    elif len({p for _, p in qp}) >= 12:
                        label = "PortScan"
                        conf_val = max(conf_val, 0.90)
                        detail = f"Hybrid(ports={len({p for _, p in qp})}/10s) " + detail
                    elif dp_int in (80, 443):
                        qh = win_http[src_ip]
                        qh.append(now_mono)
                        while qh and now_mono - qh[0] > 5.0:
                            qh.popleft()
                        if len(qh) >= 15 and label == "BENIGN":
                            if n_bytes == 0 and syn_c >= 1:
                                label = "DDoS"
                                conf_val = max(conf_val, 0.88)
                                detail = f"Hybrid(syn_flood={len(qh)}/5s) " + detail
                            else:
                                label = "DoS"
                                conf_val = max(conf_val, 0.82)
                                detail = f"Hybrid(http={len(qh)}/5s) " + detail

                is_attack = label != "BENIGN"
                n_attack += int(is_attack)
                n_benign += int(not is_attack)
                item = {
                    "time": ts_now,
                    "src": src_ip,
                    "dst": dst_ip,
                    "port": dport,
                    "pkts": f"{pkts:,}",
                    "bytes": f"{n_bytes:,}",
                    "label": label, "conf": conf_val, "detail": detail,
                    "display": True,
                    "alert": None,
                    "banner": None,
                }
                if is_attack and conf_val >= self.threshold:
                    ev = f"pkts={pkts}, bytes={n_bytes}B, SYN={syn_c}, rate={pps:.0f} pkt/s"
                    item["banner"] = f"[!] CẢNH BÁO {label} lúc {ts_now} | {src_ip} -> {dst_ip}:{dport} ({ev})"
                    item["alert"] = (
                        f"[{ts_now}] PHÁT HIỆN {label} (conf={conf_val:.2f}) | {src_ip} -> {dst_ip}:{dport}\n"
                        f"    Bằng chứng: {ev} | Model: {detail}"
                    )
                items.append(item)
            if items:
                self.event_q.put(("batch", items))

        try:
            if self.iface:
                self.event_q.put(("status", f"Đang bắt gói live trên '{self.iface}' ..."))
                sniffer = LiveFlowSniffer(iface=self.iface)
                for rows in sniffer.stream_flows(self.stop_event):
                    if self.stop_event.is_set():
                        break
                    for row in rows:
                        pending.append((0, row))
                    process_pending(force=True)
            elif self.replay_path:
                self.event_q.put(("status", f"Phat lai {self.replay_path.name} ..."))
                for row_no, row in replay_iter(self.replay_path):
                    if self.stop_event.is_set():
                        break
                    pending.append((row_no, row))
                    process_pending()
                    if self.speed:
                        time.sleep(1.0 / self.speed)
            else:
                self.event_q.put(("status", f"Doi file CSV trong {self.flow_dir} ..."))
                tails: dict[Path, FileTailer] = {}
                while not self.stop_event.is_set():
                    try:
                        current = {p for p in self.flow_dir.glob("*.csv") if p.stat().st_size > 0}
                    except OSError:
                        current = set()
                    for p in current:
                        if p not in tails:
                            tails[p] = FileTailer(p)
                    for p in list(tails):
                        if p not in current:
                            tails.pop(p)
                            continue
                        for row in tails[p].poll():
                            pending.append((0, row))
                    process_pending()
                    time.sleep(1.0)
        except Exception as e:  # noqa: BLE001
            self.event_q.put(("status", f"Loi worker: {e}"))
        process_pending(force=True)
        self.event_q.put(("done", f"Tam dung. benign={n_benign:,} attack={n_attack:,}"))


# ===========================================================================
# Ứng dụng chính
# ===========================================================================
class App(tk.Tk):
    def __init__(self, smoke: bool = False):
        super().__init__()
        self.title("NIDS — Phát hiện tấn công mạng Real-time (CIC-IDS2017)")
        self.geometry("1320x840")
        self.configure(bg=BG)
        self._setup_style()
        self.event_q: "q.Queue" = q.Queue(maxsize=4000)
        self.stop_event = threading.Event()
        self.worker = None
        self.counts: dict[str, int] = {}
        self.n_benign, self.n_attack = 0, 0
        self._photo_cm = None   # tham chiếu PhotoImage confusion matrix (tab 1)
        self._photo_eda = None  # tham chiếu PhotoImage biểu đồ EDA (tab 2)

        self.note = ttk.Notebook(self)
        self.note.pack(fill="both", expand=True, padx=8, pady=8)
        self.tab1 = ttk.Frame(self.note)
        self.tab2 = ttk.Frame(self.note)
        self.tab3 = ttk.Frame(self.note)
        self.note.add(self.tab1, text="  So sánh mô hình  ")
        self.note.add(self.tab2, text="  EDA dữ liệu  ")
        self.note.add(self.tab3, text="  Phát hiện Real-time  ")

        self._build_tab1()
        self._build_tab2()
        self._build_tab3()

        self.status = tk.StringVar(value="Sẵn sàng. Tab 3: nhấn [Bắt Live (Interface)] hoặc [Phát lại CSV...] để thử.")
        tk.Label(self, textvariable=self.status, anchor="w", bg=BG, fg=MUTED,
                 font=(FONT_UI, 9)).pack(fill="x", padx=12, pady=(0, 6))

        self.after(150, self._poll_events)
        self._refresh_tab1()
        self._refresh_tab2()
        if not HAS_MPL_TK:
            self.set_status("Chu y: matplotlib-TkAgg khong nap duoc — GUI tu dong dung Tk thuan "
                            "(pip install matplotlib de co bieu do dep hon)")
        if smoke:
            self.after(3000, self._smoke_done)

    def _setup_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        self.option_add("*Font", (FONT_UI, 9))
        style.configure("TFrame", background=BG)
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", background="#e5e7eb", foreground=MUTED,
                        padding=(16, 8), font=(FONT_UI, 10, "bold"))
        style.map("TNotebook.Tab",
                  background=[("selected", PANEL), ("active", "#f3f4f6")],
                  foreground=[("selected", FG), ("active", FG)])
        style.configure("Treeview", background=PANEL, fieldbackground=PANEL, foreground=FG,
                        rowheight=27, borderwidth=1, relief="solid", font=(FONT_UI, 9))
        style.configure("Treeview.Heading", background="#f3f4f6", foreground=FG,
                        relief="flat", font=(FONT_UI, 9, "bold"))
        style.map("Treeview.Heading", background=[("active", "#e5e7eb")])
        style.map("Treeview", background=[("selected", "#e5e7eb")],
                  foreground=[("selected", FG)])
        style.configure("TButton", background=FG, foreground="#ffffff",
                        padding=(11, 5), relief="flat", font=(FONT_UI, 9, "bold"))
        style.map("TButton", background=[("active", "#374151"), ("disabled", "#d1d5db")],
                  foreground=[("disabled", "#9ca3af")])
        style.configure("TCheckbutton", background=BG, foreground=FG, font=(FONT_UI, 9))
        style.map("TCheckbutton", background=[("active", BG)])
        style.configure("TCombobox", fieldbackground=PANEL, background="#e5e7eb",
                        foreground=FG, arrowcolor=FG)

    def _smoke_done(self):
        print("GUI SMOKE OK (HAS_MPL_TK=%s)" % HAS_MPL_TK)
        self.destroy()

    def set_status(self, text: str):
        self.status.set(text)

    # ------------------------------------------------------------------ TAB 1
    def _build_tab1(self):
        f = self.tab1
        top = tk.Frame(f, bg=BG)
        top.pack(fill="x", padx=10, pady=6)
        tk.Label(top, text="So sánh độ chính xác: TabR-lite (Deep Learning) & 5 mô hình Machine Learning",
                 bg=BG, fg=FG, font=("Segoe UI", 12, "bold")).pack(side="left")
        ttk.Button(top, text="Cập nhật", command=self._refresh_tab1).pack(side="right")

        body = tk.Frame(f, bg=BG)
        body.pack(fill="both", expand=True, padx=10, pady=4)

        # trái: biểu đồ cột
        if HAS_MPL_TK:
            self.fig_cmp = Figure(figsize=(6.4, 4.8), dpi=100, facecolor=BG)
            self.ax_cmp = self.fig_cmp.add_subplot(111)
            self.canvas_cmp = FigureCanvasTkAgg(self.fig_cmp, master=body)
            self.canvas_cmp.get_tk_widget().pack(side="left", fill="both", expand=True)
        else:
            self.canvas_bar = tk.Canvas(body, width=560, height=430, bg=PANEL,
                                        highlightthickness=0)
            self.canvas_bar.pack(side="left", fill="both", expand=True)

        # phải: bảng số liệu + confusion matrix
        right = tk.Frame(body, bg=BG)
        right.pack(side="left", fill="both", expand=True, padx=(10, 0))
        cols = ("model", "accuracy", "macro_f1", "weighted_f1")
        self.tree_metrics = ttk.Treeview(right, columns=cols, show="headings", height=7)
        for cid, txt, w, anc in zip(
            cols,
            ("Mô hình", "Accuracy", "Macro-F1", "Weighted-F1"),
            (205, 85, 85, 95),
            ("w", "center", "center", "center"),
        ):
            self.tree_metrics.heading(cid, text=txt)
            self.tree_metrics.column(cid, width=w, anchor=anc)
        self.tree_metrics.tag_configure("tabr_row", background="#e5e7eb", foreground=FG,
                                        font=(FONT_UI, 9, "bold"))
        self.tree_metrics.pack(fill="x")

        sel = tk.Frame(right, bg=BG)
        sel.pack(fill="x", pady=8)
        tk.Label(sel, text="Confusion Matrix của:", bg=BG, fg=FG,
                 font=("Segoe UI", 9, "bold")).pack(side="left")
        self.cmb_model = ttk.Combobox(sel, state="readonly", width=26)
        self.cmb_model.pack(side="left", padx=8)
        self.cmb_model.bind("<<ComboboxSelected>>", lambda _e: self._show_confusion())

        if HAS_MPL_TK:
            self.fig_cm = Figure(figsize=(4.6, 3.9), dpi=100, facecolor=BG)
            self.ax_cm = self.fig_cm.add_subplot(111)
            self.canvas_cm = FigureCanvasTkAgg(self.fig_cm, master=right)
            self.canvas_cm.get_tk_widget().pack(fill="both", expand=True)
        else:
            self.lbl_cm = tk.Label(right, bg=PANEL)
            self.lbl_cm.pack(fill="both", expand=True)

    def _refresh_tab1(self):
        metrics_path = ROOT / "reports" / "metrics.json"
        if not metrics_path.exists():
            msg = "Chua co ket qua train.\nChay:  python3 train.py"
            if HAS_MPL_TK:
                self.ax_cmp.clear()
                self.ax_cmp.text(0.5, 0.5, msg, ha="center", va="center", color=FG, fontsize=12)
                self.ax_cmp.axis("off")
                self.canvas_cmp.draw_idle()
            else:
                self.canvas_bar.delete("all")
                self.canvas_bar.create_text(280, 200, text=msg, fill=FG, font=("TkDefaultFont", 13))
            self.tree_metrics.delete(*self.tree_metrics.get_children())
            self.cmb_model["values"] = []
            return

        metrics = json.loads(metrics_path.read_text())
        names = (["tabr"] if "tabr" in metrics else []) + [n for n in metrics if n != "tabr"]
        acc = [metrics[n]["accuracy"] * 100.0 for n in names]
        mf1 = [metrics[n]["macro_f1"] * 100.0 for n in names]
        short_labels = [MODEL_SHORT.get(n, n) for n in names]
        full_labels = [MODEL_LABELS.get(n, n) for n in names]

        if HAS_MPL_TK:
            self.ax_cmp.clear()
            self.ax_cmp.set_facecolor(PANEL)
            x = np.arange(len(names))
            b1 = self.ax_cmp.bar(x - 0.19, acc, 0.38, label="Accuracy (%)", color="#111827")
            b2 = self.ax_cmp.bar(x + 0.19, mf1, 0.38, label="Macro-F1 (%)", color="#9ca3af")
            for bar in b1:
                h = bar.get_height()
                self.ax_cmp.text(bar.get_x() + bar.get_width() / 2, h + 0.35,
                                 f"{h:.2f}%", ha="center", va="bottom", color="#111827", fontsize=7.5, fontweight="bold")
            for bar in b2:
                h = bar.get_height()
                self.ax_cmp.text(bar.get_x() + bar.get_width() / 2, h + 0.35,
                                 f"{h:.2f}%", ha="center", va="bottom", color="#4b5563", fontsize=7.5)
            self.ax_cmp.set_xticks(x)
            self.ax_cmp.set_xticklabels(short_labels, fontsize=8.5)
            self.ax_cmp.set_ylim(95.0, 101.5)
            self.ax_cmp.grid(axis="y", color=GRIDC, alpha=0.6, linestyle="--")
            self.ax_cmp.legend(facecolor=PANEL, edgecolor=GRIDC, labelcolor=FG,
                               fontsize=9, loc="lower right")
            self.ax_cmp.set_title("Độ chính xác trên tập test (CIC-IDS2017, 80% Train / 20% Test)",
                                  fontsize=10.5, color=FG, fontweight="bold", pad=10)
            self.ax_cmp.tick_params(colors=FG)
            for s in self.ax_cmp.spines.values():
                s.set_color(GRIDC)
            self.fig_cmp.tight_layout()
            self.canvas_cmp.draw_idle()
        else:
            self._draw_bar_tk(self.canvas_bar, short_labels,
                              {"Accuracy": [a / 100.0 for a in acc], "Macro-F1": [m / 100.0 for m in mf1]},
                              title="Do chinh xac tren tap test")

        self.tree_metrics.delete(*self.tree_metrics.get_children())
        for n, full in zip(names, full_labels):
            tags = ("tabr_row",) if n == "tabr" else ()
            self.tree_metrics.insert("", "end", values=(
                full,
                f"{metrics[n]['accuracy']*100:.2f}%",
                f"{metrics[n]['macro_f1']*100:.2f}%",
                f"{metrics[n]['weighted_f1']*100:.2f}%",
            ), tags=tags)
        self.cmb_model["values"] = full_labels
        if full_labels:
            self.cmb_model.current(0)
            self._show_confusion()

    def _draw_bar_tk(self, canvas: tk.Canvas, names: list[str],
                     series: dict[str, list[float]], title: str = ""):
        """Vẽ biểu đồ cột nhóm bằng Tk thuần (fallback khi không có matplotlib)."""
        canvas.delete("all")
        w, h = max(canvas.winfo_width(), 400), max(canvas.winfo_height(), 300)
        n_series = len(series)
        group_w = (w - 90) / max(len(names), 1)
        bar_w = min(26, (group_w - 8) / max(n_series, 1))
        colors = [ACCENT, ORANGE, OKG, BAD]
        canvas.create_text(w // 2, 18, text=title, fill=FG, font=("TkDefaultFont", 11, "bold"))
        # trục + lưới
        x0, y0, x1, y1 = 70, 50, w - 20, h - 55
        for frac in (0.25, 0.5, 0.75, 1.0):
            yy = y1 - (y1 - y0) * frac
            canvas.create_line(x0, yy, x1, yy, fill=GRIDC, dash=(2, 3))
            canvas.create_text(x0 - 6, yy, text=f"{frac:.0%}", fill="#9fb2c8",
                               font=("TkDefaultFont", 7), anchor="e")
        canvas.create_line(x0, y0 - 6, x0, y1, fill=GRIDC)
        canvas.create_line(x0, y1, x1, y1, fill=GRIDC)
        for i, name in enumerate(names):
            gx = x0 + i * group_w + 6
            for s_idx, (s_name, vals) in enumerate(series.items()):
                v = min(max(vals[i], 0), 1)
                bh = (y1 - y0) * v
                bx = gx + s_idx * bar_w
                canvas.create_rectangle(bx, y1 - bh, bx + bar_w, y1,
                                        fill=colors[s_idx % len(colors)], outline="")
                canvas.create_text(bx + bar_w / 2, y1 - bh - 6, text=f"{v:.3f}", fill=FG,
                                   font=("TkDefaultFont", 7))
            canvas.create_text(gx + (bar_w * n_series) / 2, y1 + 14, text=name, fill=FG,
                               font=("TkDefaultFont", 9))
        for s_idx, s_name in enumerate(series):
            lx = x0 + s_idx * 110
            canvas.create_rectangle(lx, h - 32, lx + 10, h - 24,
                                    fill=colors[s_idx % len(colors)], outline="")
            canvas.create_text(lx + 14, h - 28, text=s_name, fill=FG, font=("TkDefaultFont", 8),
                               anchor="w")

    def _show_confusion(self):
        disp = self.cmb_model.get()
        name = LABEL_TO_KEY.get(disp, disp)
        png = ROOT / "reports" / f"confusion_{name}.png"
        if HAS_MPL_TK:
            self.ax_cm.clear()
            if not name or not png.exists():
                self.ax_cm.text(0.5, 0.5, "Khong tim thay\nconfusion_" + (name or "?") + ".png",
                                ha="center", va="center", color=FG)
                self.ax_cm.axis("off")
            else:
                self.ax_cm.imshow(mpimg.imread(png))
                self.ax_cm.axis("off")
            self.canvas_cm.draw_idle()
        else:
            if name and png.exists():
                self._photo_cm = tk.PhotoImage(master=self, file=str(png))
                self.lbl_cm.configure(image=self._photo_cm)
            else:
                self.lbl_cm.configure(image="", text="Khong tim thay " + png.name,
                                      fg="#9fb2c8", bg=PANEL)

    # ------------------------------------------------------------------ TAB 2
    def _build_tab2(self):
        f = self.tab2
        top = tk.Frame(f, bg=BG)
        top.pack(fill="x", padx=10, pady=6)
        tk.Label(top, text="EDA dataset CIC-IDS2017", bg=BG, fg=FG,
                 font=("TkDefaultFont", 12, "bold")).pack(side="left")
        ttk.Button(top, text="Chay phan tich EDA", command=self._run_eda).pack(side="right", padx=4)
        ttk.Button(top, text="Lam moi", command=self._refresh_tab2).pack(side="right", padx=4)

        body = tk.Frame(f, bg=BG)
        body.pack(fill="both", expand=True, padx=10, pady=4)
        left = tk.Frame(body, bg=BG)
        left.pack(side="left", fill="both", expand=True)
        tk.Label(left, text="Bieu do:", bg=BG, fg=FG).pack(anchor="w")
        self.cmb_eda = ttk.Combobox(left, state="readonly", width=40)
        self.cmb_eda.pack(anchor="w", pady=2)
        self.cmb_eda.bind("<<ComboboxSelected>>", lambda _e: self._show_eda_image())

        if HAS_MPL_TK:
            self.fig_eda = Figure(figsize=(7.4, 5.2), dpi=100, facecolor=BG)
            self.ax_eda = self.fig_eda.add_subplot(111)
            self.canvas_eda = FigureCanvasTkAgg(self.fig_eda, master=left)
            self.canvas_eda.get_tk_widget().pack(fill="both", expand=True)
        else:
            self.lbl_eda = tk.Label(left, bg=PANEL)
            self.lbl_eda.pack(fill="both", expand=True)

        right = tk.Frame(body, bg=BG)
        right.pack(side="left", fill="both", expand=True, padx=(8, 0))
        tk.Label(right, text="Thong ke tong hop:", bg=BG, fg=FG).pack(anchor="w")
        self.txt_eda = scrolledtext.ScrolledText(right, width=52, height=30, bg=PANEL,
                                                 fg=FG, insertbackground=FG, relief="flat")
        self.txt_eda.pack(fill="both", expand=True)

    def _refresh_tab2(self):
        reports = ROOT / "reports"
        pngs = sorted(p.name for p in reports.glob("eda_*.png"))
        self.cmb_eda["values"] = pngs
        if pngs:
            self.cmb_eda.current(0)
            self._show_eda_image()
        jpath = reports / "eda.json"
        self.txt_eda.delete("1.0", "end")
        if jpath.exists():
            info = json.loads(jpath.read_text())
            lines = [
                f"Thoi diem phan tich : {info.get('generated_at')}",
                f"Tong so flow       : {info.get('total_rows', 0):,}",
                f"So file (ngay)     : {len(info.get('row_counts', {}))}",
                "",
                "--- So dong theo file (ngay) ---",
            ]
            for n, c in info.get("row_counts", {}).items():
                lines.append(f"  {n:<58} {c:>12,}")
            lines += ["", "--- Phan bo nhuan goc (top 15) ---"]
            for k, v in list(info.get("label_counts", {}).items())[:15]:
                lines.append(f"  {k:<34} {v:>12,}")
            lines += ["", "--- 5 lop chien thuat (gom nhan) ---"]
            for k, v in info.get("grouped_counts", {}).items():
                lines.append(f"  {k:<34} {v:>12,}")
            lines += ["",
                      f"Mau feature phan tich : {info.get('sample_rows', 0):,}",
                      f"O NaN (trong mau)     : {info.get('nan_cells_in_sample', 0):,}",
                      f"Dong trung lap (mau)  : {info.get('duplicate_rows_in_sample', 0):,}",
                      f"Ty le mat can bang    : {info.get('imbalance_ratio', 0):.1f}x"]
            summ = info.get("numeric_summary", {})
            if summ:
                lines += ["", "--- Thong ke feature quan trong (mau) ---"]
                for k, s in summ.items():
                    lines.append(f"  {k[:30]:<32} mean={s['mean']:.2f} median={s['median']:.2f} "
                                 f"max={s['max']:.2f}")
            self.txt_eda.insert("1.0", "\n".join(lines))
        else:
            self.txt_eda.insert("1.0", "Chua co bao cao EDA.\nNhan nut 'Chay phan tich EDA' "
                                       "(can dataset trong thu muc data/).")

    def _show_eda_image(self):
        name = self.cmb_eda.get()
        png = ROOT / "reports" / name
        if HAS_MPL_TK:
            self.ax_eda.clear()
            if not name or not png.exists():
                self.ax_eda.text(0.5, 0.5, "Chua co bieu do EDA", ha="center", va="center",
                                 color=FG)
                self.ax_eda.axis("off")
            else:
                self.ax_eda.imshow(mpimg.imread(png))
                self.ax_eda.axis("off")
            self.canvas_eda.draw_idle()
        else:
            if name and png.exists():
                self._photo_eda = tk.PhotoImage(master=self, file=str(png))
                self.lbl_eda.configure(image=self._photo_eda)
            else:
                self.lbl_eda.configure(image="", text="Chua co bieu do EDA", fg="#9fb2c8",
                                       bg=PANEL)

    def _run_eda(self):
        def job():
            self.event_q.put(("status", "Dang chay EDA (doc du lieu, doan doi phut) ..."))
            try:
                from eda import generate_eda
                generate_eda(ROOT / "data", ROOT / "reports", sample_per_file=150_000)
                self.event_q.put(("eda_done", None))
                self.event_q.put(("status", "EDA hoan tat"))
            except Exception as e:  # noqa: BLE001
                self.event_q.put(("status", f"Loi EDA: {e}"))
        threading.Thread(target=job, daemon=True).start()

    # ------------------------------------------------------------------ TAB 3
    def _build_tab3(self):
        f = self.tab3
        ctrl = tk.Frame(f, bg=BG)
        ctrl.pack(fill="x", padx=10, pady=6)

        tk.Label(ctrl, text="Card mạng:", bg=BG, fg=FG, font=("Segoe UI", 9, "bold")).pack(side="left")
        self.cmb_iface = ttk.Combobox(ctrl, values=list_interfaces(), state="readonly", width=12)
        self.cmb_iface.pack(side="left", padx=4)
        self.cmb_iface.current(0)
        self.btn_live = ttk.Button(ctrl, text="Bắt Live (Interface)", command=self._start_live_iface)
        self.btn_live.pack(side="left", padx=(2, 10))

        tk.Label(ctrl, text="Thư mục CSV:", bg=BG, fg=FG).pack(side="left")
        self.var_dir = tk.StringVar(value=str(Path.home() / "nids" / "flows"))
        tk.Entry(ctrl, textvariable=self.var_dir, width=22, bg=PANEL, fg=FG,
                 insertbackground=FG, relief="flat").pack(side="left", padx=4)
        ttk.Button(ctrl, text="...", width=2, command=self._pick_dir).pack(side="left")

        self.btn_start = ttk.Button(ctrl, text="Theo dõi CSV", command=self._start_monitor)
        self.btn_start.pack(side="left", padx=6)
        self.btn_stop = ttk.Button(ctrl, text="Tạm dừng", command=self._stop_monitor,
                                   state="disabled")
        self.btn_stop.pack(side="left")
        ttk.Button(ctrl, text="Reset", command=self._reset_monitor).pack(side="left", padx=4)
        ttk.Button(ctrl, text="Phát lại CSV...", command=self._pick_replay).pack(side="left",
                                                                                 padx=4)

        tk.Label(ctrl, text="Ngưỡng:", bg=BG, fg=FG).pack(side="left", padx=(10, 2))
        self.var_threshold = tk.StringVar(value="0.35")
        ttk.Spinbox(ctrl, from_=0.1, to=0.99, increment=0.05, width=5,
                    textvariable=self.var_threshold).pack(side="left")
        self.var_show_benign = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctrl, text="Hiện flow benign", variable=self.var_show_benign).pack(
            side="left", padx=8)

        body = tk.Frame(f, bg=BG)
        body.pack(fill="both", expand=True, padx=10, pady=4)

        # trái: bảng lưu lượng real-time (chỉ hiện lưu lượng chạy, không hiện cột dự đoán)
        left = tk.Frame(body, bg=BG)
        left.pack(side="left", fill="both", expand=True)
        tk.Label(left, text="Lưu lượng mạng đang chạy (Real-time):", bg=BG, fg=FG,
                 font=(FONT_UI, 9, "bold")).pack(anchor="w")
        cols = ("time", "src", "dst", "port", "pkts", "bytes")
        self.tree_flow = ttk.Treeview(left, columns=cols, show="headings", height=18)
        for cid, txt, w in zip(cols, ("Thời gian", "Nguồn", "Đích", "Port", "Gói tin", "Dung lượng (B)"),
                               (80, 130, 130, 65, 80, 105)):
            self.tree_flow.heading(cid, text=txt)
            self.tree_flow.column(cid, width=w, anchor="w")
        vsb = ttk.Scrollbar(left, orient="vertical", command=self.tree_flow.yview)
        self.tree_flow.configure(yscrollcommand=vsb.set)
        self.tree_flow.pack(side="left", fill="both", expand=True)
        vsb.pack(side="left", fill="y")
        self.tree_flow.tag_configure("flow", background=PANEL, foreground=FG)

        # phải: banner cảnh báo tự động + log bằng chứng + bộ đếm + biểu đồ
        right = tk.Frame(body, bg=BG)
        right.pack(side="left", fill="both", expand=True, padx=(10, 0))
        self.lbl_alert_banner = tk.Label(
            right, text="Trạng thái: An toàn — Chưa phát hiện tấn công",
            bg="#f3f4f6", fg=OKG, anchor="w", padx=8, pady=5,
            font=(FONT_UI, 9, "bold"), relief="solid", borderwidth=1)
        self.lbl_alert_banner.pack(fill="x", pady=(0, 6))
        tk.Label(right, text="CẢNH BÁO TẤN CÔNG (Thời gian & Bằng chứng):", bg=BG, fg=BAD,
                 font=(FONT_UI, 10, "bold")).pack(anchor="w")
        self.txt_alert = scrolledtext.ScrolledText(right, width=58, height=10, bg="#fef2f2",
                                                   fg="#991b1b", insertbackground=FG,
                                                   relief="solid", borderwidth=1,
                                                   font=(FONT_MONO, 9))
        self.txt_alert.pack(fill="x")

        stats = tk.Frame(right, bg=BG)
        stats.pack(fill="x", pady=6)
        self.lbl_benign = tk.Label(stats, text="Benign: 0", bg=BG, fg=OKG,
                                   font=(FONT_UI, 11, "bold"))
        self.lbl_benign.pack(side="left", padx=(0, 14))
        self.lbl_attack = tk.Label(stats, text="Attack: 0", bg=BG, fg=BAD,
                                   font=(FONT_UI, 11, "bold"))
        self.lbl_attack.pack(side="left")

        if HAS_MPL_TK:
            self.fig_live = Figure(figsize=(4.8, 2.6), dpi=100, facecolor=BG)
            self.ax_live = self.fig_live.add_subplot(111)
            self.canvas_live = FigureCanvasTkAgg(self.fig_live, master=right)
            self.canvas_live.get_tk_widget().pack(fill="both", expand=True)
        else:
            self.canvas_live_bar = tk.Canvas(right, width=430, height=210, bg=PANEL,
                                             highlightthickness=0)
            self.canvas_live_bar.pack(fill="both", expand=True)

    def _pick_dir(self):
        d = filedialog.askdirectory()
        if d:
            self.var_dir.set(d)

    def _pick_replay(self):
        p = filedialog.askopenfilename(initialdir=str(ROOT / "data"),
                                       filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if not p:
            return
        self._start_worker(Path(p))
        self.set_status(f"Dang phat lai {Path(p).name} (tab: Phat hien real-time)")

    def _start_live_iface(self):
        iface = self.cmb_iface.get() or "any"
        svcs = ensure_lab_services()
        self._start_worker(None, iface=iface)
        svc_msg = f" | Dịch vụ: {', '.join(svcs)}" if svcs else ""
        self.set_status(f"Đang bắt gói trực tiếp trên '{iface}'{svc_msg}")

    def _start_monitor(self):
        svcs = ensure_lab_services()
        self._start_worker(None)
        svc_msg = f" | Dịch vụ: {', '.join(svcs)}" if svcs else ""
        self.set_status(f"Đang theo dõi {self.var_dir.get()}{svc_msg}")

    def _start_worker(self, replay: Path | None, iface: str | None = None):
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("Đang chạy", "Bộ phát hiện đang chạy. Nhấn 'Tạm dừng' trước.")
            return
        self.stop_event.clear()
        self.event_q = q.Queue(maxsize=4000)
        self.worker = DetectorWorker(
            ROOT / "models", Path(self.var_dir.get()).expanduser(), replay,
            float(self.var_threshold.get()), self.var_show_benign.get(),
            speed=0, event_q=self.event_q, stop_event=self.stop_event, iface=iface)
        self.worker.start()
        self.btn_start["state"], self.btn_live["state"], self.btn_stop["state"] = "disabled", "disabled", "normal"

    def _stop_monitor(self):
        self.stop_event.set()
        self.btn_start["state"], self.btn_live["state"], self.btn_stop["state"] = "normal", "normal", "disabled"
        self.set_status("Đang tạm dừng ...")

    def _reset_monitor(self):
        """Xóa sạch bảng lưu lượng, log cảnh báo, bộ đếm và biểu đồ để thử lại từ đầu."""
        self.n_benign, self.n_attack = 0, 0
        self.counts.clear()
        self.tree_flow.delete(*self.tree_flow.get_children())
        self.txt_alert.delete("1.0", "end")
        self.lbl_benign["text"] = "Benign: 0"
        self.lbl_attack["text"] = "Attack: 0"
        self.lbl_alert_banner.configure(
            text="Trạng thái: An toàn — Chưa phát hiện tấn công",
            bg="#f3f4f6", fg=OKG)
        self._draw_live_chart()
        self.set_status("Đã reset bộ đếm và nhật ký — sẵn sàng bắt lại từ đầu.")

    # ------------------------------------------------------------------ sự kiện
    def _poll_events(self):
        try:
            if not self.winfo_exists():
                return
        except tk.TclError:
            return
        try:
            while True:
                kind, payload = self.event_q.get_nowait()
                if kind == "batch":
                    self._handle_batch(payload)
                elif kind == "status":
                    self.set_status(payload)
                elif kind == "eda_done":
                    self._refresh_tab2()
                elif kind == "done":
                    self.set_status(payload)
                    self.btn_start["state"], self.btn_live["state"], self.btn_stop["state"] = "normal", "normal", "disabled"
        except q.Empty:
            pass
        except tk.TclError:
            return  # cửa sổ đang đóng
        self.after(150, self._poll_events)

    def _handle_batch(self, rows: list[dict]):
        try:
            self._handle_batch_inner(rows)
        except tk.TclError:
            return  # cửa sổ đang đóng

    def _handle_batch_inner(self, rows: list[dict]):
        children_added = False
        for item in rows:
            if item["label"] == "BENIGN":
                self.n_benign += 1
            else:
                self.n_attack += 1
                self.counts[item["label"]] = self.counts.get(item["label"], 0) + 1
            if item.get("alert"):
                self.lbl_alert_banner.configure(
                    text=item.get("banner") or "PHÁT HIỆN TẤN CÔNG!",
                    bg="#fee2e2", fg=BAD)
                self.txt_alert.insert("end", item["alert"] + "\n")
                self.txt_alert.see("end")
            if item.get("display"):
                self.tree_flow.insert("", 0, values=(
                    item["time"], item["src"], item["dst"], item["port"],
                    item["pkts"], item["bytes"]), tags=("flow",))
                children_added = True
        # giữ tối đa 400 dòng hiển thị
        kids = self.tree_flow.get_children()
        if len(kids) > 400:
            for k in kids[400:]:
                self.tree_flow.delete(k)
        if children_added:
            self.tree_flow.see(self.tree_flow.get_children()[0])
        self.lbl_benign["text"] = f"Benign: {self.n_benign:,}"
        self.lbl_attack["text"] = f"Attack: {self.n_attack:,}"
        self._draw_live_chart()

    def _draw_live_chart(self):
        labels = list(self.counts)
        vals = [self.counts[k] for k in labels]
        colors = [ATTACK_COLOR.get(k, BAD) for k in labels]
        if HAS_MPL_TK:
            self.ax_live.clear()
            self.ax_live.set_facecolor(PANEL)
            self.ax_live.bar(labels, vals, color=colors or [FG])
            self.ax_live.set_title("Số flow tấn công phát hiện theo loại", fontsize=9.5,
                                   color=FG, fontweight="bold")
            self.ax_live.grid(axis="y", color=GRIDC, alpha=0.5, linestyle="--")
            self.ax_live.tick_params(colors=FG, labelsize=8.5)
            for s in self.ax_live.spines.values():
                s.set_color(GRIDC)
            self.fig_live.tight_layout()
            self.canvas_live.draw_idle()
        else:
            canvas = self.canvas_live_bar
            canvas.delete("all")
            w, h = max(canvas.winfo_width(), 300), max(canvas.winfo_height(), 150)
            canvas.create_text(w // 2, 14, text="So flow tan cong phat hien theo loai",
                               fill=FG, font=("TkDefaultFont", 10, "bold"))
            if not labels:
                canvas.create_text(w // 2, h // 2 + 10, text="Chua phat hien tan cong nao",
                                   fill="#9fb2c8", font=("TkDefaultFont", 10))
                return
            vmax = max(max(vals), 1)
            y = 44
            for name, v, col in zip(labels, vals, colors):
                canvas.create_text(10, y + 8, text=name, fill=FG, font=("TkDefaultFont", 9),
                                   anchor="w")
                bw = int((w - 120) * v / vmax)
                canvas.create_rectangle(90, y, 90 + max(bw, 2), y + 16, fill=col, outline="")
                canvas.create_text(98 + max(bw, 2), y + 8, text=f"{v:,}", fill=FG,
                                   font=("TkDefaultFont", 9), anchor="w")
                y += 30


def main() -> int:
    ap = argparse.ArgumentParser(description="NIDS GUI (Kali)")
    ap.add_argument("--smoke", action="store_true", help="Mo giao dien 3 giay roi tu dong tat")
    args = ap.parse_args()
    try:
        app = App(smoke=args.smoke)
    except tk.TclError as e:
        print("Khong mo duoc Tkinter:", e)
        print("Tren Kali hay chay: sudo apt install python3-tk")
        return 1
    app.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
