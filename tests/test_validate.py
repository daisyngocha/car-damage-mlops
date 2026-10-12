import shutil
from pathlib import Path

from omegaconf import OmegaConf
from PIL import Image

from src.validate import run_checks

CLASSES = ["01-minor", "02-moderate", "03-severe"]


def make_cfg(**overrides):
    base = {
        "splits": ["train", "val"],
        "classes": CLASSES,
        "min_per_class": 1,
        "min_share": 0.0,
        "min_side": 8,
        "fail_on_conflict": True,
        "fail_on_leakage": False,
        "report_path": "unused",
    }
    return OmegaConf.create({**base, **overrides})


def make_dataset(root: Path, n: int = 2) -> Path:
    for s, split in enumerate(["train", "val"]):
        for i, cls in enumerate(CLASSES):
            d = root / split / cls
            d.mkdir(parents=True)
            for j in range(n):
                Image.new("RGB", (16, 16), (i * 80, j * 40, s * 120)).save(d / f"{j}.jpg")
    return root


def test_clean_dataset_passes(tmp_path):
    report = run_checks(make_dataset(tmp_path), make_cfg())
    assert report["errors"] == []
    assert report["counts"]["train"] == {c: 2 for c in CLASSES}


def test_unknown_class_fails(tmp_path):
    raw = make_dataset(tmp_path)
    (raw / "train" / "abc").mkdir()
    shutil.copy(raw / "train" / "01-minor" / "0.jpg", raw / "train" / "abc" / "x.jpg")
    assert any("lớp lạ" in e for e in run_checks(raw, make_cfg())["errors"])


def test_corrupt_image_fails(tmp_path):
    raw = make_dataset(tmp_path)
    (raw / "val" / "03-severe" / "broken.jpg").write_bytes(b"not an image")
    assert any("ảnh hỏng" in e for e in run_checks(raw, make_cfg())["errors"])


def test_conflict_fails_and_leakage_warns(tmp_path):
    raw = make_dataset(tmp_path)
    src = raw / "train" / "01-minor" / "0.jpg"
    shutil.copy(src, raw / "train" / "02-moderate" / "dup.jpg")  # cùng ảnh, khác nhãn
    shutil.copy(src, raw / "val" / "01-minor" / "leak.jpg")  # cùng ảnh ở train và val
    report = run_checks(raw, make_cfg())
    assert report["n_conflict_groups"] == 1
    assert report["n_leakage_groups"] == 1
    assert any("khác nhãn" in e for e in report["errors"])
    assert any("train và val" in w for w in report["warnings"])
