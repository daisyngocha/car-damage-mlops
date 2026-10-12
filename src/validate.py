"""Kiểm tra data/raw (cấu trúc ImageFolder) trước khi train.

Exit code 1 nếu có lỗi chặn; chi tiết ghi ra reports/validation.json.
"""

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import hydra
from omegaconf import DictConfig
from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def file_md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def run_checks(raw: Path, v: DictConfig) -> dict:
    allowed = set(v.classes)
    errors: list[str] = []
    warnings: list[str] = []
    counts: dict[str, Counter] = {}
    hashes: dict[str, list[str]] = defaultdict(list)

    for split in v.splits:
        split_dir = raw / split
        if not split_dir.is_dir():
            errors.append(f"thiếu thư mục {split_dir}")
            continue
        class_dirs = sorted(d for d in split_dir.iterdir() if d.is_dir())
        found = {d.name for d in class_dirs}
        if found - allowed:
            errors.append(f"{split}: lớp lạ {sorted(found - allowed)}")
        if allowed - found:
            errors.append(f"{split}: thiếu lớp {sorted(allowed - found)}")

        counts[split] = Counter()
        for cls_dir in class_dirs:
            for f in sorted(p for p in cls_dir.rglob("*") if p.is_file()):
                rel = f.relative_to(raw).as_posix()
                if f.suffix.lower() not in IMG_EXT:
                    warnings.append(f"không phải ảnh (ImageFolder bỏ qua): {rel}")
                    continue
                try:
                    with Image.open(f) as im:
                        im.verify()
                    with Image.open(f) as im:
                        width, height = im.size
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"ảnh hỏng: {rel} ({type(exc).__name__})")
                    continue
                if min(width, height) < v.min_side:
                    warnings.append(f"ảnh nhỏ {width}x{height}: {rel}")
                counts[split][cls_dir.name] += 1
                hashes[file_md5(f)].append(rel)

        total = sum(counts[split].values())
        for cls in sorted(allowed & found):
            n = counts[split][cls]
            if n < v.min_per_class:
                errors.append(f"{split}/{cls}: chỉ có {n} ảnh (< {v.min_per_class})")
            elif total and n / total < v.min_share:
                warnings.append(f"{split}/{cls}: chiếm {n / total:.1%} (< {v.min_share:.0%})")

    dups = [paths for paths in hashes.values() if len(paths) > 1]
    conflicts = [p for p in dups if len({x.split("/")[1] for x in p}) > 1]
    leakage = [p for p in dups if len({x.split("/")[0] for x in p}) > 1]
    for name, groups, fail in [
        ("trùng ảnh khác nhãn", conflicts, v.fail_on_conflict),
        ("trùng ảnh giữa train và val", leakage, v.fail_on_leakage),
    ]:
        if groups:
            (errors if fail else warnings).append(f"{len(groups)} nhóm {name}, ví dụ {groups[0]}")

    return {
        "raw_dir": str(raw),
        "counts": {s: dict(sorted(c.items())) for s, c in counts.items()},
        "n_duplicate_groups": len(dups),
        "n_conflict_groups": len(conflicts),
        "n_leakage_groups": len(leakage),
        "duplicate_groups": dups,
        "errors": errors,
        "warnings": warnings,
    }


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    v = cfg.data.validation
    report = run_checks(Path(cfg.data.root), v)
    out = Path(v.report_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    slim = {
        **report,
        "duplicate_groups": report["duplicate_groups"][:20],
        "n_warnings": len(report["warnings"]),
        "warnings": report["warnings"][:50],
    }
    out.write_text(json.dumps(slim, indent=2, ensure_ascii=False) + "\n")

    for split, c in report["counts"].items():
        print(split, c)
    for msg in report["errors"]:
        print(f"[LỖI] {msg}")
    print(f"{len(report['errors'])} lỗi, {len(report['warnings'])} cảnh báo → {out}")
    sys.exit(1 if report["errors"] else 0)


if __name__ == "__main__":
    main()
