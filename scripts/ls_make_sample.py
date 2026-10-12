"""Lấy mẫu ảnh val để tự gán nhãn trên Label Studio; nhãn gốc ghi riêng vào answer_key.csv."""

import csv
import random
import shutil
from pathlib import Path

RAW = Path("data/raw/val")
OUT = Path("data/ls_sample")
KEY = Path("data/annotations/answer_key.csv")
N_PER_CLASS = 20

random.seed(42)
OUT.mkdir(parents=True, exist_ok=True)
KEY.parent.mkdir(parents=True, exist_ok=True)

rows = []
for cls_dir in sorted(p for p in RAW.iterdir() if p.is_dir()):
    files = sorted(cls_dir.iterdir())
    rows += [(f, cls_dir.name) for f in random.sample(files, N_PER_CLASS)]
random.shuffle(rows)

with KEY.open("w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(["sample_id", "true_label", "source_path"])
    for i, (f, label) in enumerate(rows, 1):
        sample_id = f"s{i:03d}{f.suffix.lower()}"
        shutil.copy2(f, OUT / sample_id)
        writer.writerow([sample_id, label, f.as_posix()])
print(f"{len(rows)} ảnh → {OUT}, đáp án → {KEY}")
