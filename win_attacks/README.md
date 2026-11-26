# Script giả lập tấn công (chạy trên WINDOWS, gửi về Kali qua host-only)

> CHỈ dùng trong mạng lab host-only (VMnet1) của chính bạn, phục vụ demo/học tập.
> Không sử dụng chống lại hệ thống của người khác — vi phạm pháp luật.

## Chuẩn bị trên Windows

```
pip install paramiko requests scapy
```
- Cài [Npcap](https://npcap.com) (chọn "WinPcap API-compatible Mode") nếu dùng syn_flood.py.
- Cài [Zenmap/NCrack](https://nmap.org/download.html) nếu muốn dùng bản GUI/CLI của nmap.

## Bảng map lớp tấn công -> script

| Lớp model học | Script | Cú pháp mẫu |
|---|---|---|
| PortScan | `port_scan.py` | `python port_scan.py --target 192.168.56.20 --end 1024 --threads 64` |
| BruteForce (SSH) | `ssh_bruteforce.py` | `python ssh_bruteforce.py --target 192.168.56.20 --user ubuntu --passwords 123456,kali,root` |
| BruteForce (FTP) | `ftp_bruteforce.py` | `python ftp_bruteforce.py --target 192.168.56.20 --user ftpuser --passwords 123456,ftp` |
| DDoS (SYN flood) | `syn_flood.py` | `python syn_flood.py --target 192.168.56.20 --port 80 --threads 4` (chạy bằng Administrator) |
| DoS (HTTP flood) | `http_flood.py` | `python http_flood.py --target http://192.168.56.20 --threads 50` |

Trên Kali cần có sẵn dịch vụ đích: SSH (`sudo systemctl start ssh`), FTP (vsftpd),
web server (Apache/nginx). IP của Windows trong mạng host-only là IP card
"VMware Network Adapter VMnet1".

## Kịch bản demo gợi ý (mỗi đợt 30-60 giây, xen kẽ traffic bình thường)

1. Chỉ duyệt web/ping bình thường 2-3 phút -> model phải báo BENIGN.
2. `port_scan.py` 30s -> GUI phải bấm cờ PortScan.
3. Nghỉ 1 phút -> BENIGN.
4. `ssh_bruteforce.py` 30s -> BruteForce.
5. `http_flood.py` 30s -> DoS.
6. `syn_flood.py` 30s -> DDoS.
