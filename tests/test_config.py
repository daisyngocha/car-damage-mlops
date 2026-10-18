from pathlib import Path

from hydra import compose, initialize_config_dir

CONF_DIR = str(Path(__file__).resolve().parents[1] / "conf")


def load(overrides=()):
    with initialize_config_dir(config_dir=CONF_DIR, version_base=None):
        return compose(config_name="config", overrides=list(overrides))


def test_default_config_composes():
    cfg = load()
    assert cfg.train.epochs > 0 and cfg.train.lr > 0


def test_override():
    assert load(["train.lr=0.005"]).train.lr == 0.005
