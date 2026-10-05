# Baseline v0 — pytorch/examples imagenet/main.py nguyên trạng

- Ngày: 2026-10-xx
- Code: commit <hash>, upstream pytorch/examples@4206858 (imagenet, 2025-07-07), BSD-3-Clause
- Data: Kaggle prajwalbhamere/car-damage-severity-dataset — License: <...>; train = <n> ảnh, val = <n> ảnh
- Lệnh: python main.py -a resnet18 --pretrained --epochs 3 -b 32 --lr 0.01 -j 2 --seed 42 data/raw
- Môi trường: WSL2 Ubuntu 22.04, Python 3.11, torch 2.14.1+cpu, torchvision 0.29.1+cpu
- Kết quả (val, epoch cuối): Acc@1 = <...>   (Acc@5 bỏ qua — head 1000 lớp)
- Thời gian: <...> / 3 epoch
- Tái lập: run1 vs run2 → <giống / lệch bao nhiêu>
- Gap ghi nhận: head 1000 lớp, top-5 vô nghĩa, checkpoint ghi ra thư mục gốc, metric chỉ print, code distributed gây nhiễu
