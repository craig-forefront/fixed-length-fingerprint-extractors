"""
scripts/train_500k.py trains with or without the localization network
(learned alignment); nothing else differs between the two.
"""

import importlib.util
import sys
from os.path import abspath, dirname, join

import cv2
import numpy as np
import pytest

from flx.models.deep_print_loss import LossWeights
from flx.setup import config

SCRIPTS_DIR = join(dirname(dirname(abspath(__file__))), "scripts")
VARIANTS = ["LocTexMinu", "LocTex", "TexMinu", "Tex"]


@pytest.fixture(scope="module")
def train_500k():
    spec = importlib.util.spec_from_file_location(
        "train_500k", join(SCRIPTS_DIR, "train_500k.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeExtractor:
    def __init__(self):
        self.fit_args = None

    def fit(self, **kwargs):
        self.fit_args = kwargs


def _run_main(train_500k, tmp_path, monkeypatch, extra_args, with_minutiae=False):
    """
    Runs main() with fake model constructors. Returns the name of the variant
    that was built, the arguments of its constructor and those of fit.
    """
    image_dir = tmp_path / "images"
    image_dir.mkdir(exist_ok=True)
    minutiae_dir = tmp_path / "minutiae"
    minutiae_dir.mkdir(exist_ok=True)
    for subject in range(1, 5):
        for impression in (1, 2):
            name = f"{subject}_{impression}"
            cv2.imwrite(str(image_dir / f"{name}.png"), np.zeros((8, 8), dtype=np.uint8))
            (minutiae_dir / f"{name}.mnt").write_text("1\n8 8\n4 4 0.5\n")
    built = []
    extractor = _FakeExtractor()
    for variant in VARIANTS:

        def constructor(*args, _variant=variant, **kwargs):
            built.append((_variant, args, kwargs))
            return extractor

        monkeypatch.setattr(train_500k, "get_DeepPrint_" + variant, constructor)
    argv = [
        "train_500k.py",
        "--dataset_dir", str(image_dir),
        "--output_dir", str(tmp_path / "model"),
        "--num_subjects", "4",
        "--num_impressions", "2",
        "--validation_split", "0",
    ]
    if with_minutiae:
        argv += ["--minutiae_dir", str(minutiae_dir)]
    monkeypatch.setattr(sys, "argv", argv + extra_args)
    train_500k.main()
    assert len(built) == 1
    return built[0][0], built[0][2], extractor.fit_args


@pytest.mark.parametrize(
    "localization, with_minutiae, name",
    [
        (True, True, "LocTexMinu"),
        (True, False, "LocTex"),
        (False, True, "TexMinu"),
        (False, False, "Tex"),
    ],
)
def test_variant_name(train_500k, localization, with_minutiae, name):
    assert train_500k.variant_name(localization, with_minutiae) == name


@pytest.mark.parametrize("with_minutiae", [True, False])
def test_option_switches_the_localization_network_on_and_off(
    train_500k, tmp_path, monkeypatch, with_minutiae
):
    on, _, _ = _run_main(
        train_500k, tmp_path, monkeypatch, ["--localization", "on"], with_minutiae
    )
    off, _, _ = _run_main(
        train_500k, tmp_path, monkeypatch, ["--localization", "off"], with_minutiae
    )
    assert on == ("LocTexMinu" if with_minutiae else "LocTex")
    assert off == ("TexMinu" if with_minutiae else "Tex")


def test_default_is_the_documented_one(train_500k, tmp_path, monkeypatch):
    assert train_500k.DEFAULT_LOCALIZATION in ("on", "off")
    name, _, _ = _run_main(train_500k, tmp_path, monkeypatch, [])
    assert name.startswith("Loc") == (train_500k.DEFAULT_LOCALIZATION == "on")
    assert f"the default" in train_500k.__doc__


@pytest.mark.parametrize("with_minutiae", [True, False])
def test_both_ways_get_the_same_everything_else(
    train_500k, tmp_path, monkeypatch, with_minutiae
):
    common = [
        "--learning_rate", "0.002", "--warmup_steps", "7", "--num_epochs", "3",
        "--patience", "2", "--embedding_dim", "32", "--loss_weight_center", "0.5",
    ]
    _, built_on, fit_on = _run_main(
        train_500k, tmp_path, monkeypatch, common + ["--localization", "on"], with_minutiae
    )
    _, built_off, fit_off = _run_main(
        train_500k, tmp_path, monkeypatch, common + ["--localization", "off"], with_minutiae
    )
    assert built_on == built_off
    assert built_on["loss_weights"] == LossWeights(center=0.5)
    assert fit_on.keys() == fit_off.keys()
    for key in ("learning_rate", "num_epochs", "patience", "out_dir"):
        assert fit_on[key] == fit_off[key]
    for key in ("fingerprints", "minutia_maps", "labels"):
        assert list(fit_on[key].ids) == list(fit_off[key].ids)
        assert type(fit_on[key].data_loader) is type(fit_off[key].data_loader)


def test_learning_rate_factor_of_the_localization_network(
    train_500k, tmp_path, monkeypatch
):
    _, _, fit = _run_main(train_500k, tmp_path, monkeypatch, ["--localization", "on"])
    assert (
        fit["learning_rate"].localization_factor
        == config.LOCALIZATION_LEARNING_RATE_FACTOR
    )
    _, _, fit = _run_main(
        train_500k,
        tmp_path,
        monkeypatch,
        ["--localization", "on", "--localization_lr_factor", "0.2"],
    )
    assert fit["learning_rate"].localization_factor == 0.2


def test_invalid_values_are_rejected(train_500k, tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        _run_main(train_500k, tmp_path, monkeypatch, ["--localization", "maybe"])
    with pytest.raises(SystemExit):
        _run_main(
            train_500k, tmp_path, monkeypatch, ["--localization_lr_factor", "0"]
        )


def test_summary_names_the_variant(train_500k, tmp_path, monkeypatch, capsys):
    _run_main(train_500k, tmp_path, monkeypatch, ["--localization", "off"])
    out = capsys.readouterr().out
    assert "without Learned Alignment (Tex" in out
    assert "--localization off" in out
    _run_main(train_500k, tmp_path, monkeypatch, ["--localization", "on"])
    out = capsys.readouterr().out
    assert "with Learned Alignment (LocTex" in out
    assert "learning rate of the localization network times 0.035" in out
