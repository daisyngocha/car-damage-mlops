# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Car damage severity classification (3 classes: minor / moderate / severe) from images. It is an MLOps learning project, refactored from upstream `pytorch/examples/imagenet/main.py` (BSD-3-Clause). The project is split into milestones (G3, G4, G6, ...), which are referenced in comments and in `docs/baseline.md`. Docs and code comments are in Vietnamese.

Note: the lower half of `README.md` (the "ImageNet training in PyTorch" section, with `main.py` and distributed flags) is leftover upstream text. `main.py` no longer exists. The current entry points are `src.train` and `src.evaluate`.

## Commands

```bash
uv sync --locked                       # Python 3.11, CPU-only torch 2.14.1 from the pytorch-cpu index
uv run pytest -q                       # all tests (CI runs: uv run pytest tests -v)
uv run pytest tests/test_pipeline.py::test_evaluate_writes_metrics_and_confusion   # single test
uv run ruff check --fix . && uv run ruff format .   # lint/format (line-length 100)
uv run pre-commit run --all-files

uv run python -m src.train             # train → models/model.pt + reports/train_*.{csv,json}
uv run python -m src.evaluate          # eval  → reports/metrics.json + reports/confusion.csv
uv run python -m src.train train.epochs=1 data.num_workers=0 device=cpu   # Hydra overrides
uv run python -m src.train -m train.lr=0.01,0.001                         # sweep → multirun/
```

- Add dependencies with `uv add <pkg>` (or `uv add --dev <pkg>`) and always commit `pyproject.toml` and `uv.lock` together. The `uv-lock` pre-commit hook checks that they stay in sync.
- A pre-commit hook (`no-commit-to-branch`) blocks commits on `main`, so do the work on a branch and open a PR.
- `torch`/`torchvision` versions are pinned on purpose: the baseline Acc@1 of 60.081 was measured with them. Don't bump them casually.

## Architecture

- **Config (Hydra)**: `conf/config.yaml` composes the `data/`, `model/` and `train/` groups. `hydra.job.chdir: false` keeps the cwd at the repo root, so relative paths (`data/raw`, `models`, `reports`) resolve from there. Each run writes its logs to `outputs/<date>/<time>/`.
- **src/ modules**: `data.py` builds ImageFolder datasets, transforms and loaders. `model.py` builds the model with `torchvision.models.get_model` and swaps `fc` for an `num_classes` head when `replace_head`. `engine.py` holds `train_one_epoch` and `predict` (which returns `y_true`/`y_pred`). `utils.py` holds the seed and device helpers. `train.py` and `evaluate.py` are the `@hydra.main` entry points.
- **Checkpoint contract**: `train.py` saves only the best val-acc epoch to `models/model.pt` as `{state_dict, classes, epoch, val_acc, config}`. `evaluate.py` rebuilds the model from the **config stored in the checkpoint**, not the current config. It sets `weights=None` before loading the state_dict, and it fails if the data classes differ from `ckpt["classes"]` or if the checkpoint has `replace_head=false`.
- **Parity with upstream**: `model.replace_head=false` keeps the 1000-class head so a run can be checked against the v0 baseline (`docs/baseline.md`). `train.py` deliberately creates the model *before* the data loaders so the random sequence matches upstream. Keep that order.
- **Reports**: `reports/` is committed. `confusion.csv` (one row per val image) is shaped for the DVC plots planned in G6.
- **Data**: `data/raw/{train,val}/<class>/` (Kaggle Car Damage Severity Dataset, CC BY-NC-SA 4.0). It is not committed.

## Tests

`tests/conftest.py` builds a real Hydra config from `conf/` using `compose()` with overrides that shrink the run: a tiny generated ImageFolder in `tmp_path`, `weights=null`, CPU, 1 epoch. Tests call `train.main(cfg)` and `evaluate.main(cfg)` directly with that config. Pipeline tests must override `paths.*` to `tmp_path` so they never overwrite the real `models/` or `reports/`.
