#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_dataset.py — Tải dataset CIC-IDS2017 (thư mục MachineLearningCVE, 8 file CSV).

Nguồn ưu tiên:
  1. Mirror Hugging Face (c01dsnap/CIC-IDS2017) — tải trực tiếp, không cần tài khoản.
  2. Kaggle official mirror (cicdataset/cicids2017) — dùng kagglehub, cần ~/.kaggle/kaggle.json
     hoặc biến môi trường KAGGLE_USERNAME/KAGGLE_KEY (tùy chọn).
  3. Server gốc của UNB (cicresearch.ca) — hiện BẮT BUỘC điền form trên trình duyệt:
     mở https://www.unb.ca/cic/datasets/ids-2017.html -> kéo xuống dưới -> điền form ->
     tải 8 file CSV rồi chép vào thư mục data/ của project.

Cách chạy:
    python3 download_dataset.py                 # tải tất cả file còn thiếu vào ./data
    python3 download_dataset.py --data-dir /duong/dan
    python3 download_dataset.py --only Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
"""
import argparse
import os
import sys
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
DATALAD: str = "https://huggingface.co/datasets/c01dsnap/CIC-IDS2017/resolve/main"
KAGGLE_SLUG: str = "cicdataset/cicids2017"
UNB_FORM_URL: str = "https://www.unb.ca/cic/datasets/ids-2017.html"

# Tên file + kích thước byte của mirror HF (dùng để kiểm tra tính toàn vẹn).
FILES: dict[str, int] = {
    "Monday-WorkingHours.pcap_ISCX.csv": 176927918,
    "Tuesday-WorkingHours.pcap_ISCX.csv": 135078995,
    "Wednesday-workingHours.pcap_ISCX.csv": 225166395,
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv": 52023263,
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv": 83102436,
    "Friday-WorkingHours-Morning.pcap_ISCX.csv": 58316725,
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv": 77123859,
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv": 76906168,
}
CSV_HEADER_TOKEN = "Destination Port"  # cột đầu tiên của mọi file MachineLearningCVE
# ---------------------------------------------------------------------------


def human(nbytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024 or unit == "GB":
            return f"{nbytes:,.1f} {unit}" if unit != "B" else f"{int(nbytes)} B"
        nbytes /= 1024
    return f"{nbytes:,.1f} GB"


def looks_valid(path: Path) -> bool:
    """File đã tải hợp lệ nếu đủ lớn và dòng đầu chứa cột header chuẩn."""
    if not path.exists() or path.stat().st_size < 1_000_000:
        return False
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return CSV_HEADER_TOKEN in f.readline()
    except OSError:
        return False


def download_from_hf(name: str, dest: Path, session: requests.Session) -> bool:
    url = f"{DATALAD}/{name}"
    tmp = dest.with_suffix(".part")
    headers = {}
    # Resume nếu có file .part còn dang dở
    if tmp.exists() and tmp.stat().st_size < FILES.get(name, 0):
        headers["Range"] = f"bytes={tmp.stat().st_size}-"
    try:
        with session.get(url, headers=headers, stream=True, timeout=(15, 60), allow_redirects=True) as r:
            if r.status_code not in (200, 206):
                print(f"    [!] HTTP {r.status_code} từ Hugging Face")
                return False
            mode = "ab" if r.status_code == 206 else "wb"
            total = int(r.headers.get("content-length", 0)) + (tmp.stat().st_size if r.status_code == 206 else 0)
            done = tmp.stat().st_size if r.status_code == 206 else 0
            t0 = time.time()
            with open(tmp, mode) as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
                        done += len(chunk)
                        if total:
                            speed = done / max(time.time() - t0, 1e-9)
                            pct = min(100.0, 100.0 * done / total)
                            print(f"\r    {pct:5.1f}%  {human(done)}/{human(total)}  {human(speed)}/s",
                                  end="", flush=True)
            print()
    except requests.RequestException as e:
        print(f"    [!] Lỗi mạng: {e}")
        return False
    if not looks_valid(tmp):
        print("    [!] File tải về không hợp lệ (header sai hoặc quá nhỏ)")
        tmp.unlink(missing_ok=True)
        return False
    # Kiểm tra kích thước nếu mirror vẫn giữ nguyên số byte gốc
    expected = FILES.get(name)
    if expected and abs(tmp.stat().st_size - expected) > 1024:
        print(f"    [~] Kích thước lệch nhẹ so với gốc ({human(tmp.stat().st_size)} "
              f"vs {human(expected)}) — vẫn chấp nhận vì header hợp lệ")
    tmp.replace(dest)
    return True


def download_from_kaggle(data_dir: Path) -> bool:
    """Dùng kagglehub (cần tài khoản Kaggle: đặt KAGGLE_USERNAME/KAGGLE_KEY)."""
    try:
        import kagglehub  # noqa
    except ImportError:
        print("[kaggle] Chưa cài kagglehub — bỏ qua nguồn Kaggle")
        return False
    have_cred = os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY") \
        or (Path.home() / ".kaggle" / "kaggle.json").exists()
    if not have_cred:
        print("[kaggle] Không có credential Kaggle — bỏ qua nguồn Kaggle")
        return False
    try:
        print(f"[kaggle] Tải dataset {KAGGLE_SLUG} (có thể mất vài phút)...")
        src = Path(kagglehub.dataset_download(KAGGLE_SLUG))
        found = 0
        for csv in src.rglob("*.csv"):
            if csv.parent.name == "MachineLearningCVE" and csv.name in FILES:
                dest = data_dir / csv.name
                if not looks_valid(dest):
                    dest.write_bytes(csv.read_bytes())
                found += 1
        print(f"[kaggle] Chép được {found}/8 file từ MachineLearningCVE")
        return found == len(FILES)
    except Exception as e:  # noqa: BLE001
        print(f"[kaggle] Lỗi: {e}")
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Tải dataset CIC-IDS2017 (MachineLearningCVE)")
    ap.add_argument("--data-dir", default=str(Path(__file__).resolve().parent / "data"))
    ap.add_argument("--only", default=None, help="Chỉ tải 1 file cụ thể (tên file)")
    ap.add_argument("--source", choices=["auto", "hf", "kaggle"], default="auto")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    targets = list(FILES) if not args.only else [args.only]
    for t in targets:
        if t not in FILES:
            print(f"[!] Không biết file '{t}'. Danh sách hợp lệ:\n    " + "\n    ".join(FILES))
            return 1

    missing = [f for f in targets if not looks_valid(data_dir / f)]
    already = len(targets) - len(missing)
    if already:
        print(f"[✓] {already} file đã có sẵn trong {data_dir}")
    if not missing:
        print("[✓] Đủ dataset, không cần tải thêm. Chạy: python3 train.py")
        return 0

    session = requests.Session()
    session.headers["User-Agent"] = "nids-kali/1.0 (educational IDS project)"

    if args.source in ("auto", "hf"):
        print(f"[hf ] Tải {len(missing)} file từ Hugging Face mirror ...")
        for name in missing:
            print(f"    -> {name} ({human(FILES[name])})")
            ok = download_from_hf(name, data_dir / name, session)
            if ok:
                print(f"    [✓] {name}")
        missing = [f for f in missing if not looks_valid(data_dir / f)]
        if not missing:
            print("[✓] Hoàn tất tải dataset. Chạy: python3 train.py")
            return 0

    if args.source in ("auto", "kaggle"):
        if download_from_kaggle(data_dir):
            missing = [f for f in missing if not looks_valid(data_dir / f)]
        missing = [f for f in missing if not looks_valid(data_dir / f)]
        if not missing:
            print("[✓] Hoàn tất tải dataset. Chạy: python3 train.py")
            return 0

    if missing:
        print(f"\n[!] Vẫn còn {len(missing)} file chưa tải được:")
        for m in missing:
            print("    - " + m)
        print("\nHướng dẫn tải thủ công (nguồn gốc):")
        print(f"  1. Mở {UNB_FORM_URL}")
        print("  2. Kéo xuống phần download, điền form (mất ~30 giây)")
        print("  3. Tải thư mục MachineLearningCVE, chép 8 file .csv vào: " + str(data_dir))
        print("  4. Chạy lại lệnh này để kiểm tra, rồi chạy: python3 train.py")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
