# NIDS-Kali — Phần mềm phát hiện tấn công mạng REAL-TIME trên Kali Linux

Phần mềm Python huấn luyện từ dataset **CIC-IDS2017** bằng **6 model**
(Random Forest, Decision Tree, KNN, Linear Logistic Regression, XGBoost và **TabR-lite**
— mô hình học sâu attention cho dữ liệu bảng), sau đó **bắt lưu lượng live**
trên mạng host-only VMware (Windows tấn công -> Kali) và **cảnh báo tấn công ra
màn hình theo thời gian thực**.

Có giao diện đồ họa 3 tab: so sánh độ chính xác các model — EDA dữ liệu — phát hiện
tấn công real-time với khung lưu lượng đến.

```
┌─ Windows (máy thật, attacker) ────────┐       ┌─ Kali VM (VMware) ─────────────────────────┐
│ win_attacks/: port_scan, brute force, │ VMnet1│ ens33: web/SSH/FTP server                  │
│ syn_flood, http_flood                 │ host- │ CICFlowMeter (bắt live) -> CSV flow        │
│                                       │ only  │ realtime_detector.py / nids_gui.py         │
│                                       │       │   -> ensemble 6 model -> CẢNH BÁO màn hình │
└───────────────────────────────────────┘       └────────────────────────────────────────────┘
```

## 1. Cấu trúc thư mục

| File | Chức năng |
|---|---|
| `download_dataset.py` | Tải 8 file CSV CIC-IDS2017 (mirror Hugging Face, không cần tài khoản) |
| `train.py` | Làm sạch -> cân bằng 5 lớp -> train 6 model -> lưu model + báo cáo |
| `eda.py` | Phân tích EDA, sinh biểu đồ + `reports/eda.json` |
| `tabr_model.py` | Triển khai TabR-lite (PyTorch): transformer tabular + nhánh kNN |
| `realtime_detector.py` | Detector real-time giao diện dòng lệnh (tail CSV CICFlowMeter) |
| `nids_gui.py` | **Giao diện đồ họa 3 tab** (so sánh model / EDA / real-time) |
| `win_attacks/` | Script giả lập tấn công chạy trên Windows (port scan, brute force, flood) |
| `data/` | Dataset (được tải vào đây) |
| `models/` | Model đã train (`*.joblib`, `tabr.pt`, scaler, danh sách feature) |
| `reports/` | Classification report, confusion matrix PNG, EDA PNG, metrics.json |

## 2. Cài đặt trên Kali Linux

```bash
# Python 3.9+ (Kali có sẵn). Tkinter cho GUI:
sudo apt update && sudo apt install -y python3-tk python3-venv

cd ~/NIDS-Kali
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# (khuyến nghị) bản PyTorch CPU nhẹ hơn:
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

> Nếu không cài được PyTorch, vẫn chạy được toàn bộ phần mềm — chỉ thiếu model TabR
> (5 model còn lại hoạt động bình thường).

## 3. Tải dataset + huấn luyện

```bash
python3 download_dataset.py          # tải 8 file CSV (~800 MB) vào data/
python3 train.py                     # train đầy đủ: 6 model, cap 20.000 mẫu/lớp
```

Tuỳ chọn hữu ích:

```bash
python3 train.py --quick             # thử pipeline nhanh (mẫu ít, 2 epoch TabR)
python3 train.py --cap 10000         # mỗi lớp tối đa 10.000 mẫu
python3 train.py --skip-tabr         # bỏ qua deep learning
python3 train.py --models rf,xgb     # chỉ train model chọn
```

Sau khi train xong: model nằm ở `models/`, báo cáo ở `reports/` (kèm confusion
matrix PNG và `metrics.json` dùng cho tab so sánh trong GUI).

**Kết quả đã train sẵn trong package này** (5 lớp cân bằng, cap 6.000/lớp,
test 4.180 flow — chạy lại `python3 train.py --cap 20000` để có model mạnh hơn):

| Model | Accuracy | Macro-F1 |
|---|---|---|
| XGBoost | 99.90% | 99.90% |
| Random Forest | 99.81% | 99.81% |
| Decision Tree | 99.64% | 99.64% |
| KNN | 99.57% | 99.58% |
| Linear Logistic Regression | 99.23% | 99.25% |
| **TabR-lite (deep learning)** | 88.06% | 88.32% |

> TabR-lite train trên CPU chỉ 10 epoch nên chưa hội tụ; tăng `--tabr-epochs 30`
> (hoặc chạy GPU) sẽ lên ~95%+. Ensemble trong real-time gộp cả 6 model.

**5 lớp model học** (gom từ 15 nhãn gốc, đã cân bằng cap mỗi lớp):

| Lớp | Nhãn gốc trong dataset |
|---|---|
| `BENIGN` | BENIGN |
| `DoS` | DoS Hulk, DoS GoldenEye, DoS slowloris, DoS Slowhttptest |
| `DDoS` | DDoS |
| `PortScan` | PortScan |
| `BruteForce` | FTP-Patator, SSH-Patator |

## 4. Giao diện đồ họa (3 tab)

```bash
python3 nids_gui.py
```

1. **So sánh mô hình** — biểu đồ cột Accuracy/Macro-F1 của rf, dt, knn, logreg,
   xgb, tabr + bảng số liệu + confusion matrix từng model (chọn ở combobox).
2. **EDA dữ liệu** — bấm "Chạy phân tích EDA" (hoặc đã sinh sẵn sau train):
   phân bố nhãn gốc và 5 lớp gộp, benign/attack theo từng ngày, histogram feature
   quan trọng, ma trận tương quan, bảng thống kê.
3. **Phát hiện real-time** — điền thư mục CSV live (mặc định `~/nids/flows`),
   bấm **Bật đầu**. Khung bên trái hiển thị lưu lượng đến theo thời gian thực
   (thời gian, nguồn, đích, port, dự đoán, độ tin cậy), khung cảnh báo đỏ bên
   phải + bộ đếm + biểu đồ số flow tấn công theo loại. Nút **"Phát lại CSV..."**
   cho phép test không cần CICFlowMeter (chọn 1 file CSV trong `data/`).

Chế độ dòng lệnh tương đương (nhẹ hơn, chạy SSH):

```bash
python3 realtime_detector.py --dir ~/nids/flows
python3 realtime_detector.py --replay data/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv --show-benign
```

## 5. Chạy real-time với CICFlowMeter (bắt lưu lượng thật)

CICFlowMeter là công cụ của UNB trích **cùng bộ 78 feature** như dataset, chạy
live trên interface mạng:

```bash
sudo apt install -y default-jre

# Cách 1 (khuyên dùng): tải jar từ SourceForge -> https://sourceforge.net/projects/cicflowmeter/
java -jar CICFlowMeter-4.0.jar

# Cách 2: build từ mã nguồn
git clone https://github.com/CanadianInstituteForCybersecurity/CICFlowMeter.git
cd CICFlowMeter && ./gradlew build
```

Trong GUI của CICFlowMeter:
- **Network Interface**: chọn interface host-only của Kali (thường là `ens33`)
- **Active Timeout**: đặt **5** giây (flow được flush nhanh -> cảnh báo nhanh)
- **Output Directory**: chọn `~/nids/flows` (khớp với `--dir` của detector)
- Bấm **Start Capture** — mỗi flow sẽ được ghi tiếp vào file CSV.

Rồi trên Kali: `python3 nids_gui.py` -> tab real-time -> **Bắt đầu**.

## 6. Cấu hình mạng host-only (VMware)

1. Kali VM -> Settings -> Network Adapter -> **Host-only** (VMnet1).
2. Trên Windows, card **"VMware Network Adapter VMnet1"** đặt IP tĩnh,
   ví dụ `192.168.56.1/24`.
3. Kali đặt IP tĩnh cùng subnet, ví dụ `192.168.56.20/24`
   (`sudo ip addr add 192.168.56.20/24 dev ens33` hoặc cấu hình qua NetworkManager).
4. `ping` hai chiều để kiểm tra trước khi chạy tấn công.

## 7. Giả lập tấn công từ Windows

Chuẩn bị: `pip install paramiko requests scapy`, cài [Npcap](https://npcap.com)
(chọn WinPcap API-compatible), chạy CMD/PowerShell **quyền Administrator** cho
`syn_flood.py`. Chi tiết xem `win_attacks/README.md`.

```bash
cd win_attacks
python port_scan.py      --target 192.168.56.20 --end 1024 --threads 64     # PortScan
python ssh_bruteforce.py --target 192.168.56.20 --user ubuntu --passwords 123456,kali,root
python http_flood.py     --target http://192.168.56.20 --threads 50         # DoS
python syn_flood.py      --target 192.168.56.20 --port 80 --threads 4       # DDoS
python ftp_bruteforce.py --target 192.168.56.20 --user ftpuser --passwords 123456,ftp
```

Trên Kali cần mở dịch vụ đích: `sudo systemctl start ssh`, cài vsftpd/Apache.
Kịch bản demo xen kẽ traffic bình thường — xem `win_attacks/README.md`.

## 8. Khắc phục sự cố

| Vấn đề | Cách xử lý |
|---|---|
| `ImportError: libtk8.6` khi mở GUI | `sudo apt install -y python3-tk` |
| Không có model trong GUI | Chạy `python3 train.py` trước (tab so sánh cần `reports/metrics.json`) |
| Tải dataset lỗi mạng | Chạy lại `download_dataset.py` (có resume); hoặc tải thủ công qua form UNB (script sẽ in hướng dẫn) |
| Cảnh báo "feature không có trong CSV" | Đầu ra CICFlowMeter phải đủ 80 cột; kiểm tra version CICFlowMeter |
| Cảnh báo dồn dập khi traffic bình thường | Tăng `--threshold` (vd 0.7), hoặc tune thêm: bỏ `Destination Port` khi train |
| Accuracy 99.99% bất thường | Dataset có nhiều dòng trùng — pipeline đã drop duplicate; vẫn nên kiểm tra confusion matrix |
| TabR chậm trên CPU | `--tabr-epochs 6` hoặc dùng `--skip-tabr`; TabR chạy tốt nhất với GPU |

## 9. Ghi chú học thuật + pháp lý

- Dataset: Iman Sharafaldin, Arash Habibi Lashkari, Ali A. Ghorbani, *"Toward
  Generating a New Intrusion Detection Dataset and Intrusion Traffic
  Characterization"*, ICISSP 2018. Khi công bố kết quả, hãy trích dẫn paper này.
- TabR-lite là bản tái hiện đơn giản lấy cảm hứng từ Gorishniy et al. 2024
  (*"TabR: Untangling Transformer and Feature Attribution in Interacting with
  Tabular Data"*) — phục vụ demo/giảng dạy, không phải code gốc của Yandex.
- Script trong `win_attacks/` **chỉ dùng trong mạng lab host-only của chính bạn**
  để demo hệ thống phát hiện. Sử dụng chống lại hệ thống khác là vi phạm pháp luật.
- Độ trễ phát hiện: model dựa trên flow, cần đợi flow flush (5s active timeout)
  — quy mô phát hiện là giây, không phải millisecond.
