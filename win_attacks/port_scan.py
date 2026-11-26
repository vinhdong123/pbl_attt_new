#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
port_scan.py — Giả lập PortScan từ Windows vào victim (Kali) trên mạng host-only.

Chỉ dùng trong LAB của chính bạn (mạng host-only VMnet1). Không dùng vào hệ thống khác.

Ví dụ:
    python port_scan.py --target 192.168.56.20 --start 1 --end 1024 --threads 64
    python port_scan.py --target 192.168.56.20 --ports 21,22,80,443,8080
"""
import argparse
import socket
import sys
import threading
import queue


def worker(host, q: "queue.Queue[int]", open_ports: list, lock: threading.Lock, timeout: float):
    while True:
        try:
            port = q.get_nowait()
        except queue.Empty:
            return
        try:
            with socket.create_connection((host, port), timeout=timeout):
                with lock:
                    open_ports.append(port)
        except OSError:
            pass
        finally:
            q.task_done()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="IP máy nạn nhân (Kali) trong mạng host-only")
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=1000)
    ap.add_argument("--ports", default=None, help="Danh sách port cụ thể, vd 21,22,80")
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--timeout", type=float, default=0.8)
    args = ap.parse_args()

    ports = ([int(p) for p in args.ports.split(",")] if args.ports
             else list(range(args.start, args.end + 1)))
    q: "queue.Queue[int]" = queue.Queue()
    for p in ports:
        q.put(p)

    open_ports: list[int] = []
    lock = threading.Lock()
    threads = [threading.Thread(target=worker, args=(args.target, q, open_ports, lock, args.timeout),
                                daemon=True) for _ in range(max(1, args.threads))]
    print(f"[scan] TCP connect scan {args.target} với {len(ports)} port, {args.threads} threads ...")
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"[scan] Xong. Port mở: {sorted(open_ports)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
