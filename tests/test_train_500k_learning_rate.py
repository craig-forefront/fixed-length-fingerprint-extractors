"""
Tests of the learning rate options of scripts/train_500k.py
"""
import importlib.util
import sys
from os.path import abspath, dirname, join

import cv2
import numpy as np
import pytest

from flx.models.learning_rate import LearningRateConfig
from flx.setup import config

SCRIPTS_DIR = join(dirname(dirname(abspath(__file__))), "scripts")


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


def _run_main(train_500k, tmp_path, monkeypatch, extra_args) -> LearningRateConfig:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for subject in range(1, 5):
        for impression in (1, 2):
            cv2.imwrite(
                str(image_dir / f"{subject}_{impression}.png"),
                np.zeros((8, 8), dtype=np.uint8),
            )
    extractor = _FakeExtractor()
    monkeypatch.setattr(train_500k, "get_DeepPrint_LocTex", lambda *a, **k: extractor)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_500k.py",
            "--dataset_dir", str(image_dir),
            "--output_dir", str(tmp_path / "model"),
            "--num_subjects", "4",
            "--num_impressions", "2",
            "--validation_split", "0",
        ]
        + extra_args,
    )
    train_500k.main()
    return extractor.fit_args["learning_rate"]


def test_learning_rate_options_reach_the_training(train_500k, tmp_path, monkeypatch):
    lr = _run_main(
        train_500k,
        tmp_path,
        monkeypatch,
        [
            "--learning_rate", "0.003",
            "--warmup_steps", "250",
            "--lr_schedule", "cosine",
            "--gradient_clip_norm", "2.5",
        ],
    )
    assert lr.learning_rate == 0.003
    assert lr.warmup_steps == 250
    assert lr.schedule == "cosine"
    assert lr.gradient_clip_norm == 2.5


def test_options_that_are_not_given_use_the_config(train_500k, tmp_path, monkeypatch):
    lr = _run_main(train_500k, tmp_path, monkeypatch, [])
    assert lr.learning_rate == config.LEARNING_RATE
    assert lr.warmup_steps == config.LEARNING_RATE_WARMUP_STEPS
    assert lr.schedule == config.LEARNING_RATE_SCHEDULE
    assert lr.gradient_clip_norm == config.GRADIENT_CLIP_NORM


def test_gradient_clip_norm_zero_switches_the_limit_off(
    train_500k, tmp_path, monkeypatch
):
    monkeypatch.setattr(config, "GRADIENT_CLIP_NORM", 5.0)
    lr = _run_main(train_500k, tmp_path, monkeypatch, ["--gradient_clip_norm", "0"])
    assert lr.gradient_clip_norm is None
