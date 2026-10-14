"""Chuẩn hóa ảnh raw → processed, giữ cấu trúc split/<lớp>/."""

import shutil
from pathlib import Path

import hydra
from omegaconf import DictConfig
from PIL import Image, ImageOps

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    src, dst = Path(cfg.data.raw_dir), Path(cfg.data.processed_dir)
    shutil.rmtree(dst, ignore_errors=True)
    n = 0
    for split in ("train", "val"):
        for f in sorted((src / split).rglob("*")):
            if f.suffix.lower() not in EXTS:
                continue
            out = (dst / f.relative_to(src)).with_suffix(".jpg")
            out.parent.mkdir(parents=True, exist_ok=True)
            img = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
            w, h = img.size
            s = cfg.data.resize / min(w, h)
            if s < 1:  # chỉ thu nhỏ, không phóng to
                img = img.resize((round(w * s), round(h * s)), Image.BICUBIC)
            img.save(out, quality=95)
            n += 1
    print(f"prepared {n} images -> {dst}")


if __name__ == "__main__":
    main()
