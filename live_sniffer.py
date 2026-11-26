#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
live_sniffer.py — Bắt gói tin mạng trực tiếp (live) và trích xuất 78 đặc trưng
CIC-IDS2017 theo từng flow trong thời gian thực (không cần CICFlowMeter Java).

Hoạt động trực tiếp trên Kali Linux / VMware bằng raw socket AF_PACKET
(cần quyền root: `sudo python3 nids_gui.py` hoặc `sudo python3 realtime_detector.py --iface eth0`).
Trên Windows, thử dùng Scapy nếu đã cài (`pip install scapy`).
"""
from __future__ import annotations

import math
import os
import socket
import struct
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Iterator

import numpy as np


def list_interfaces() -> list[str]:
    """Liệt kê các card mạng trên hệ thống (ưu tiên eth0/ens33/wlan0/any)."""
    ifaces: list[str] = ["any"]
    try:
        names = [name for _, name in socket.if_nameindex()]
        for n in names:
            if n not in ifaces and n != "lo":
                ifaces.append(n)
    except Exception:
        pass
    if sys.platform.startswith("linux") and os.path.isdir("/sys/class/net"):
        try:
            for n in sorted(os.listdir("/sys/class/net")):
                if n not in ifaces and n != "lo":
                    ifaces.append(n)
        except OSError:
            pass
    return ifaces


@dataclass
class _FlowState:
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    proto: int
    start_ts: float
    last_ts: float
    fwd_times: list[float] = field(default_factory=list)
    bwd_times: list[float] = field(default_factory=list)
    all_times: list[float] = field(default_factory=list)
    fwd_lens: list[int] = field(default_factory=list)
    bwd_lens: list[int] = field(default_factory=list)
    all_lens: list[int] = field(default_factory=list)
    fwd_hdr_len: int = 0
    bwd_hdr_len: int = 0
    fwd_psh: int = 0
    bwd_psh: int = 0
    fwd_urg: int = 0
    bwd_urg: int = 0
    fin_cnt: int = 0
    syn_cnt: int = 0
    rst_cnt: int = 0
    psh_cnt: int = 0
    ack_cnt: int = 0
    urg_cnt: int = 0
    cwe_cnt: int = 0
    ece_cnt: int = 0
    init_win_fwd: int = -1
    init_win_bwd: int = -1
    act_data_pkt_fwd: int = 0
    min_seg_size_fwd: int = 0
    closed: bool = False


def _stats(vals: list[float] | list[int]) -> tuple[float, float, float, float, float]:
    """Trả về (total, min, max, mean, std) nhanh gọn."""
    if not vals:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    arr = np.asarray(vals, dtype=np.float64)
    return (
        float(arr.sum()),
        float(arr.min()),
        float(arr.max()),
        float(arr.mean()),
        float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
    )


def _iat_stats(times: list[float]) -> tuple[float, float, float, float, float]:
    """Tính IAT (microseconds) -> (total, min, max, mean, std)."""
    if len(times) < 2:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    diffs = [(times[i] - times[i - 1]) * 1_000_000.0 for i in range(1, len(times))]
    return _stats(diffs)


def _active_idle_stats(times: list[float], idle_thresh_s: float = 5.0) -> tuple[list[float], list[float]]:
    if len(times) < 2:
        return [], []
    actives, idles = [], []
    act_start = times[0]
    prev = times[0]
    for t in times[1:]:
        gap = t - prev
        if gap >= idle_thresh_s:
            actives.append((prev - act_start) * 1_000_000.0)
            idles.append(gap * 1_000_000.0)
            act_start = t
        prev = t
    if prev > act_start:
        actives.append((prev - act_start) * 1_000_000.0)
    return actives, idles


def flow_to_row(fl: _FlowState) -> dict:
    """Chuyển _FlowState thành dict 78 đặc trưng khớp models/features.json."""
    dur_us = max((fl.last_ts - fl.start_ts) * 1_000_000.0, 1.0)
    dur_s = dur_us / 1_000_000.0

    n_fwd = len(fl.fwd_lens)
    n_bwd = len(fl.bwd_lens)
    n_all = n_fwd + n_bwd

    fwd_tot, fwd_min, fwd_max, fwd_mean, fwd_std = _stats(fl.fwd_lens)
    bwd_tot, bwd_min, bwd_max, bwd_mean, bwd_std = _stats(fl.bwd_lens)
    all_tot, all_min, all_max, all_mean, all_std = _stats(fl.all_lens)

    _, flow_iat_min, flow_iat_max, flow_iat_mean, flow_iat_std = _iat_stats(fl.all_times)
    fwd_iat_tot, fwd_iat_min, fwd_iat_max, fwd_iat_mean, fwd_iat_std = _iat_stats(fl.fwd_times)
    bwd_iat_tot, bwd_iat_min, bwd_iat_max, bwd_iat_mean, bwd_iat_std = _iat_stats(fl.bwd_times)

    actives, idles = _active_idle_stats(fl.all_times)
    _, act_min, act_max, act_mean, act_std = _stats(actives)
    _, idl_min, idl_max, idl_mean, idl_std = _stats(idles)

    return {
        "Timestamp": time.strftime("%H:%M:%S", time.localtime(fl.last_ts)),
        "Source IP": fl.src_ip,
        "Destination IP": fl.dst_ip,
        "Destination Port": fl.dst_port,
        "Flow Duration": dur_us,
        "Total Fwd Packets": n_fwd,
        "Total Backward Packets": n_bwd,
        "Total Length of Fwd Packets": fwd_tot,
        "Total Length of Bwd Packets": bwd_tot,
        "Fwd Packet Length Max": fwd_max,
        "Fwd Packet Length Min": fwd_min,
        "Fwd Packet Length Mean": fwd_mean,
        "Fwd Packet Length Std": fwd_std,
        "Bwd Packet Length Max": bwd_max,
        "Bwd Packet Length Min": bwd_min,
        "Bwd Packet Length Mean": bwd_mean,
        "Bwd Packet Length Std": bwd_std,
        "Flow Bytes/s": all_tot / dur_s,
        "Flow Packets/s": n_all / dur_s,
        "Flow IAT Mean": flow_iat_mean,
        "Flow IAT Std": flow_iat_std,
        "Flow IAT Max": flow_iat_max,
        "Flow IAT Min": flow_iat_min,
        "Fwd IAT Total": fwd_iat_tot,
        "Fwd IAT Mean": fwd_iat_mean,
        "Fwd IAT Std": fwd_iat_std,
        "Fwd IAT Max": fwd_iat_max,
        "Fwd IAT Min": fwd_iat_min,
        "Bwd IAT Total": bwd_iat_tot,
        "Bwd IAT Mean": bwd_iat_mean,
        "Bwd IAT Std": bwd_iat_std,
        "Bwd IAT Max": bwd_iat_max,
        "Bwd IAT Min": bwd_iat_min,
        "Fwd PSH Flags": fl.fwd_psh,
        "Bwd PSH Flags": fl.bwd_psh,
        "Fwd URG Flags": fl.fwd_urg,
        "Bwd URG Flags": fl.bwd_urg,
        "Fwd Header Length": fl.fwd_hdr_len,
        "Bwd Header Length": fl.bwd_hdr_len,
        "Fwd Packets/s": n_fwd / dur_s,
        "Bwd Packets/s": n_bwd / dur_s,
        "Min Packet Length": all_min,
        "Max Packet Length": all_max,
        "Packet Length Mean": all_mean,
        "Packet Length Std": all_std,
        "Packet Length Variance": all_std * all_std,
        "FIN Flag Count": fl.fin_cnt,
        "SYN Flag Count": fl.syn_cnt,
        "RST Flag Count": fl.rst_cnt,
        "PSH Flag Count": fl.psh_cnt,
        "ACK Flag Count": fl.ack_cnt,
        "URG Flag Count": fl.urg_cnt,
        "CWE Flag Count": fl.cwe_cnt,
        "ECE Flag Count": fl.ece_cnt,
        "Down/Up Ratio": (n_bwd / n_fwd) if n_fwd > 0 else 0.0,
        "Average Packet Size": all_tot / max(n_all, 1),
        "Avg Fwd Segment Size": fwd_mean,
        "Avg Bwd Segment Size": bwd_mean,
        "Fwd Header Length.1": fl.fwd_hdr_len,
        "Fwd Avg Bytes/Bulk": 0.0,
        "Fwd Avg Packets/Bulk": 0.0,
        "Fwd Avg Bulk Rate": 0.0,
        "Bwd Avg Bytes/Bulk": 0.0,
        "Bwd Avg Packets/Bulk": 0.0,
        "Bwd Avg Bulk Rate": 0.0,
        "Subflow Fwd Packets": n_fwd,
        "Subflow Fwd Bytes": fwd_tot,
        "Subflow Bwd Packets": n_bwd,
        "Subflow Bwd Bytes": bwd_tot,
        "Init_Win_bytes_forward": fl.init_win_fwd if fl.init_win_fwd >= 0 else 0,
        "Init_Win_bytes_backward": fl.init_win_bwd if fl.init_win_bwd >= 0 else 0,
        "act_data_pkt_fwd": fl.act_data_pkt_fwd,
        "min_seg_size_forward": fl.min_seg_size_fwd,
        "Active Mean": act_mean,
        "Active Std": act_std,
        "Active Max": act_max,
        "Active Min": act_min,
        "Idle Mean": idl_mean,
        "Idle Std": idl_std,
        "Idle Max": idl_max,
        "Idle Min": idl_min,
    }


def parse_eth_packet(raw: bytes) -> tuple[str, str, int, int, int, int, int, int, int] | None:
    """Giải mã frame Ethernet -> (src_ip, dst_ip, proto, sport, dport, payload_len, hdr_len, tcp_flags, win)."""
    if len(raw) < 34:
        return None
    eth_proto = struct.unpack("!H", raw[12:14])[0]
    off = 14
    if eth_proto == 0x8100 and len(raw) >= 38:  # 802.1Q VLAN
        eth_proto = struct.unpack("!H", raw[16:18])[0]
        off = 18
    if eth_proto != 0x0800 or len(raw) < off + 20:  # Chỉ xử lý IPv4
        return None

    ver_ihl, _, total_len, _, _, _, proto, _ = struct.unpack("!BBHHHBBH", raw[off:off + 12])
    ihl = (ver_ihl & 0x0F) * 4
    if ihl < 20 or len(raw) < off + ihl:
        return None
    src_ip = socket.inet_ntoa(raw[off + 12:off + 16])
    dst_ip = socket.inet_ntoa(raw[off + 16:off + 20])
    l4 = off + ihl

    if proto == 6 and len(raw) >= l4 + 20:  # TCP
        sport, dport, _, _, off_flags, win = struct.unpack("!HHIIHH", raw[l4:l4 + 16])
        tcp_hdr_len = ((off_flags >> 12) & 0x0F) * 4
        tcp_flags = off_flags & 0xFF
        payload_len = max(0, total_len - ihl - tcp_hdr_len)
        return src_ip, dst_ip, proto, sport, dport, payload_len, tcp_hdr_len, tcp_flags, win
    if proto == 17 and len(raw) >= l4 + 8:  # UDP
        sport, dport, udp_len, _ = struct.unpack("!HHHH", raw[l4:l4 + 8])
        payload_len = max(0, udp_len - 8)
        return src_ip, dst_ip, proto, sport, dport, payload_len, 8, 0, 0
    return None


class LiveFlowSniffer:
    """Bắt gói trực tiếp trên interface và gom thành flow CIC-IDS2017."""

    def __init__(self, iface: str = "any", idle_timeout: float = 0.8, active_timeout: float = 2.0):
        self.iface = iface
        self.idle_timeout = idle_timeout
        self.active_timeout = active_timeout
        self.flows: dict[tuple, _FlowState] = {}

    def _add_packet(self, ts: float, pkt: tuple[str, str, int, int, int, int, int, int, int]) -> None:
        src_ip, dst_ip, proto, sport, dport, pay_len, hdr_len, flags, win = pkt
        fwd_key = (src_ip, sport, dst_ip, dport, proto)
        bwd_key = (dst_ip, dport, src_ip, sport, proto)

        if fwd_key in self.flows:
            fl = self.flows[fwd_key]
            is_fwd = True
        elif bwd_key in self.flows:
            fl = self.flows[bwd_key]
            is_fwd = False
        else:
            fl = _FlowState(src_ip, dst_ip, sport, dport, proto, ts, ts)
            self.flows[fwd_key] = fl
            is_fwd = True

        fl.last_ts = ts
        fl.all_times.append(ts)
        fl.all_lens.append(pay_len)

        if is_fwd:
            fl.fwd_times.append(ts)
            fl.fwd_lens.append(pay_len)
            fl.fwd_hdr_len += hdr_len
            if fl.init_win_fwd < 0:
                fl.init_win_fwd = win
            if fl.min_seg_size_fwd == 0 or hdr_len < fl.min_seg_size_fwd:
                fl.min_seg_size_fwd = hdr_len
            if pay_len > 0:
                fl.act_data_pkt_fwd += 1
        else:
            fl.bwd_times.append(ts)
            fl.bwd_lens.append(pay_len)
            fl.bwd_hdr_len += hdr_len
            if fl.init_win_bwd < 0:
                fl.init_win_bwd = win

        if proto == 6:
            if flags & 0x01:
                fl.fin_cnt += 1
                fl.closed = True
            if flags & 0x02:
                fl.syn_cnt += 1
            if flags & 0x04:
                fl.rst_cnt += 1
                fl.closed = True
            if flags & 0x08:
                fl.psh_cnt += 1
                if is_fwd:
                    fl.fwd_psh += 1
                else:
                    fl.bwd_psh += 1
            if flags & 0x10:
                fl.ack_cnt += 1
            if flags & 0x20:
                fl.urg_cnt += 1
                if is_fwd:
                    fl.fwd_urg += 1
                else:
                    fl.bwd_urg += 1
            if flags & 0x40:
                fl.ece_cnt += 1
            if flags & 0x80:
                fl.cwe_cnt += 1

    def _pop_expired(self, now: float, force: bool = False) -> list[dict]:
        ready = []
        for k in list(self.flows):
            fl = self.flows[k]
            idle = now - fl.last_ts
            active = now - fl.start_ts
            if force or (fl.closed and idle >= 0.1) or idle >= self.idle_timeout or active >= self.active_timeout:
                self.flows.pop(k)
                ready.append(flow_to_row(fl))
        return ready

    def stream_flows(self, stop_event: threading.Event) -> Iterator[list[dict]]:
        """Generator liên tục trả về danh sách các flow vừa hoàn thành."""
        if sys.platform.startswith("linux") and hasattr(socket, "AF_PACKET"):
            yield from self._stream_af_packet(stop_event)
        else:
            yield from self._stream_scapy(stop_event)

    def _stream_af_packet(self, stop_event: threading.Event) -> Iterator[list[dict]]:
        try:
            sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.ntohs(0x0003))
        except PermissionError as e:
            raise PermissionError("Cần quyền root để bắt gói live: chạy `sudo python3 ...`") from e
        if self.iface and self.iface != "any":
            sock.bind((self.iface, 0))
        sock.settimeout(0.25)
        last_flush = time.time()
        try:
            while not stop_event.is_set():
                try:
                    raw, _ = sock.recvfrom(65535)
                    pkt = parse_eth_packet(raw)
                    if pkt is not None:
                        self._add_packet(time.time(), pkt)
                except socket.timeout:
                    pass
                now = time.time()
                if now - last_flush >= 0.35:
                    last_flush = now
                    rows = self._pop_expired(now)
                    if rows:
                        yield rows
            rows = self._pop_expired(time.time(), force=True)
            if rows:
                yield rows
        finally:
            sock.close()

    def _stream_scapy(self, stop_event: threading.Event) -> Iterator[list[dict]]:
        try:
            from scapy.all import IP, TCP, UDP, sniff  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "Trên Windows cần cài `pip install scapy` (và Npcap) để bắt gói live; "
                "trên Kali Linux dùng trực tiếp raw socket không cần cài thêm."
            ) from e

        def _on_pkt(p):
            if IP not in p:
                return
            ip = p[IP]
            if TCP in p:
                l4 = p[TCP]
                hdr_len = int(l4.dataofs or 5) * 4
                pay_len = max(0, len(l4.payload))
                pkt = (ip.src, ip.dst, 6, int(l4.sport), int(l4.dport),
                       pay_len, hdr_len, int(l4.flags), int(l4.window))
                self._add_packet(time.time(), pkt)
            elif UDP in p:
                l4 = p[UDP]
                pay_len = max(0, len(l4.payload))
                pkt = (ip.src, ip.dst, 17, int(l4.sport), int(l4.dport),
                       pay_len, 8, 0, 0)
                self._add_packet(time.time(), pkt)

        kwargs = {"prn": _on_pkt, "store": False, "timeout": 1.0}
        if self.iface and self.iface != "any":
            kwargs["iface"] = self.iface
        while not stop_event.is_set():
            sniff(**kwargs)
            rows = self._pop_expired(time.time())
            if rows:
                yield rows
        rows = self._pop_expired(time.time(), force=True)
        if rows:
            yield rows


# ---------------------------------------------------------------------------
# Tự động bật dịch vụ lab (SSH:22, FTP:21, HTTP:80) khi chạy real-time trên Kali
# ---------------------------------------------------------------------------
_LAB_SERVERS: list[object] = []


def _port_open(port: int, host: str = "127.0.0.1") -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.35):
            return True
    except OSError:
        return False


def _start_dummy_tcp_server(port: int, banner: bytes, reply: bytes | None = None, delay_s: float = 0.0) -> bool:
    """Mở server TCP nhẹ trên port (nếu service hệ thống chưa cài) để tránh RST."""
    try:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", port))
        srv.listen(64)
        srv.settimeout(1.0)
    except OSError:
        return False

    stop_flag = threading.Event()

    def _handle(conn: socket.socket):
        try:
            conn.settimeout(2.0)
            conn.sendall(banner)
            if reply is not None:
                for step in range(4):
                    data = conn.recv(512)
                    if not data:
                        break
                    if delay_s > 0:
                        time.sleep(delay_s)
                    if port == 21 and step >= 1:
                        conn.sendall(b"530 Login incorrect\r\n")
                    else:
                        conn.sendall(reply)
        except OSError:
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def _accept_loop():
        while not stop_flag.is_set():
            try:
                conn, _ = srv.accept()
                threading.Thread(target=_handle, args=(conn,), daemon=True).start()
            except socket.timeout:
                continue
            except OSError:
                break

    threading.Thread(target=_accept_loop, daemon=True).start()
    _LAB_SERVERS.append((srv, stop_flag))
    return True


def _start_dummy_http_server(port: int = 80) -> bool:
    class _QuietHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<h1>NIDS Lab Server</h1>")

        def do_POST(self):
            self.do_GET()

        def log_message(self, format, *args):
            pass

    try:
        httpd = ThreadingHTTPServer(("0.0.0.0", port), _QuietHandler)
    except OSError:
        return False
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    _LAB_SERVERS.append(httpd)
    return True


def ensure_lab_services() -> list[str]:
    """Tự động bật SSH (22), FTP (21), HTTP (80) nếu đang tắt (ưu tiên service Kali, fallback Python)."""
    if not sys.platform.startswith("linux"):
        return []
    started: list[str] = []
    services = [
        (22, ["ssh", "sshd"], "SSH(22)"),
        (21, ["vsftpd", "proftpd"], "FTP(21)"),
        (80, ["apache2", "nginx"], "HTTP(80)"),
    ]
    for port, names, label in services:
        if _port_open(port):
            started.append(f"{label}:OK")
            continue
        for svc in names:
            for cmd in (["systemctl", "start", svc], ["service", svc, "start"]):
                try:
                    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=4)
                except Exception:
                    pass
                if _port_open(port):
                    break
            if _port_open(port):
                started.append(f"{label}:SVC")
                break
        if not _port_open(port):
            ok = False
            if port == 80:
                ok = _start_dummy_http_server(80)
            elif port == 22:
                ok = _start_dummy_tcp_server(22, b"SSH-2.0-OpenSSH_9.6p1 Kali-Lab\r\n")
            elif port == 21:
                ok = _start_dummy_tcp_server(21, b"220 Kali Lab FTP Server\r\n", b"331 Password required\r\n", delay_s=0.25)
            if ok:
                started.append(f"{label}:PY")
    return started


def stop_lab_services() -> None:
    """Đóng các fallback Python server nếu đã tự mở."""
    while _LAB_SERVERS:
        item = _LAB_SERVERS.pop()
        try:
            if isinstance(item, tuple):
                srv, flag = item
                flag.set()
                srv.close()
            else:
                item.shutdown()
                item.server_close()
        except Exception:
            pass
