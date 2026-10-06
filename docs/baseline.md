# Baseline v0 — pytorch/examples imagenet/main.py nguyên trạng

- Ngày: 2026-10-05
- Code: commit 209c4d1, upstream pytorch/examples@4206858 (imagenet, 2025-07-07), BSD-3-Clause
- Data: Kaggle prajwalbhamere/car-damage-severity-dataset — License: CC BY-NC-SA 4.0 (chỉ dùng phi thương mại, học cá nhân); train = 1383 ảnh, val = 248 ảnh
- Lệnh: python main.py -a resnet18 --pretrained --epochs 3 -b 32 --lr 0.01 -j 2 --seed 42 data/raw
- Môi trường: WSL2 Ubuntu 22.04, Python 3.11, torch 2.14.1+cpu, torchvision 0.29.1+cpu, CPU 8 luồng
- Kết quả val Acc@1 theo epoch: 48.790 → 57.258 → **60.081** (Acc@5 = 100 bỏ qua — head 1000 lớp)
- Mốc so sánh: đoán toàn bộ là lớp đông nhất (03-severe) = 91/248 = 36.7% → baseline hơn ~23 điểm
- Thời gian: 5m27s / 3 epoch (~2.2 s/batch)
- Tái lập: run1 vs run2 trùng khớp hoàn toàn (diff rỗng)
- Gap ghi nhận: head 1000 lớp (loss khởi đầu 9.77), top-5 vô nghĩa, checkpoint ghi ra thư mục gốc, metric chỉ print, code distributed gây nhiễu

## Ghi nguồn dữ liệu
"Car Damage Severity Dataset" — prajwalbhamere, Kaggle:
https://www.kaggle.com/datasets/prajwalbhamere/car-damage-severity-dataset
License: CC BY-NC-SA 4.0 — https://creativecommons.org/licenses/by-nc-sa/4.0/
Thay đổi: đổi tên thư mục `training` → `train`, `validation` → `val`; không chỉnh sửa ảnh hay nhãn.
