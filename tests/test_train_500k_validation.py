"""
Tests of the hold-out of validation subjects in scripts/train_500k.py
"""
import importlib.util
import random
import sys
from os.path import abspath, dirname, join

import cv2
import numpy as np
import pytest

from flx.data.dataset import IdentifierSet

SCRIPTS_DIR = join(dirname(dirname(abspath(__file__))), "scripts")


@pytest.fixture(scope="module")
def train_500k():
    spec = importlib.util.spec_from_file_location(
        "train_500k", join(SCRIPTS_DIR, "train_500k.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_split_holds_out_whole_subjects(train_500k):
    training_ids, validation_ids = train_500k.create_training_and_validation_ids(
        num_subjects=50, num_impressions=4, validation_split=0.1
    )
    train_subjects = {i.subject for i in training_ids}
    val_subjects = {i.subject for i in validation_ids}
    assert len(val_subjects) == 5
    assert len(train_subjects) == 45
    assert train_subjects.isdisjoint(val_subjects)
    assert train_subjects | val_subjects == set(range(50))
    # Every subject has all its impressions in its own set
    assert len(training_ids) == 45 * 4
    assert len(validation_ids) == 5 * 4


def test_split_is_the_same_for_the_same_seed(train_500k):
    first = train_500k.split_subjects(1000, 0.1)
    assert train_500k.split_subjects(1000, 0.1) == first
    assert train_500k.split_subjects(1000, 0.1, seed=42) == first
    other = train_500k.split_subjects(1000, 0.1, seed=7)
    assert other[1] != first[1]
    assert len(other[1]) == len(first[1]) == 100


def test_default_split_holds_out_the_same_subjects_as_before(train_500k):
    # The script used random.seed(42) and random.shuffle of the subject list
    state = random.getstate()
    subjects = list(range(1000))
    random.seed(42)
    random.shuffle(subjects)
    random.setstate(state)
    _, val_subjects = train_500k.split_subjects(1000, 0.1)
    assert val_subjects == sorted(subjects[:100])


def test_split_does_not_change_the_global_random_state(train_500k):
    random.seed(123)
    expected = random.random()
    random.seed(123)
    _, validation_ids = train_500k.create_training_and_validation_ids(20, 3, 0.2)
    train_500k.create_validation_benchmark(validation_ids, 3)
    assert random.random() == expected


def test_split_respects_the_limit_of_validation_subjects(train_500k):
    train_subjects, val_subjects = train_500k.split_subjects(
        1000, 0.1, max_validation_subjects=30
    )
    assert len(val_subjects) == 30
    assert len(train_subjects) == 970
    # A limit above the share changes nothing
    assert train_500k.split_subjects(
        1000, 0.1, max_validation_subjects=500
    ) == train_500k.split_subjects(1000, 0.1)


def test_split_of_zero_holds_out_nothing(train_500k):
    training_ids, validation_ids = train_500k.create_training_and_validation_ids(
        10, 2, 0.0
    )
    assert validation_ids is None
    assert training_ids.num_subjects == 10


def test_split_rejects_shares_that_cannot_be_validated(train_500k):
    with pytest.raises(ValueError):
        train_500k.split_subjects(100, 1.0)
    with pytest.raises(ValueError):
        train_500k.split_subjects(100, -0.1)
    # 10 % of 15 subjects is one subject: no pair of different subjects
    with pytest.raises(ValueError):
        train_500k.split_subjects(15, 0.1)
    # One impression per subject: no pair of the same subject
    with pytest.raises(ValueError):
        train_500k.create_training_and_validation_ids(100, 1, 0.1)


def test_validation_benchmark_is_the_same_for_the_same_seed(train_500k):
    _, validation_ids = train_500k.create_training_and_validation_ids(200, 3, 0.1)

    def pairs(seed):
        benchmark = train_500k.create_validation_benchmark(
            validation_ids, 3, seed=seed
        )
        return [(c.sample1, c.sample2) for c in benchmark._comparisons]

    assert pairs(42) == pairs(42)
    assert pairs(42) != pairs(7)
    comparisons = train_500k.create_validation_benchmark(validation_ids, 3)._comparisons
    # 20 subjects: 3 mated pairs and 10 non-mated pairs for each
    assert sum(c.mated for c in comparisons) == 20 * 3
    assert sum(not c.mated for c in comparisons) == 20 * 10
    val_subjects = {i.subject for i in validation_ids}
    for c in comparisons:
        assert {c.sample1.subject, c.sample2.subject} <= val_subjects


class _FakeExtractor:
    def __init__(self):
        self.fit_args = None

    def fit(self, **kwargs):
        self.fit_args = kwargs


@pytest.mark.parametrize(
    "extra_args, num_validation_subjects",
    [([], 2), (["--validation_split", "0"], 0), (["--validation_split", "0.4", "--max_validation_subjects", "3"], 3)],
)
def test_model_has_one_class_per_training_subject(
    train_500k, tmp_path, monkeypatch, extra_args, num_validation_subjects
):
    # 25 subjects with the default share of 0.1: 2 validation subjects and 23
    # training subjects. The script built the model for int(25 * 0.9) = 22.
    num_subjects = 25
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for subject in range(1, num_subjects + 1):
        for impression in (1, 2):
            cv2.imwrite(
                str(image_dir / f"{subject}_{impression}.png"),
                np.zeros((8, 8), dtype=np.uint8),
            )

    extractor = _FakeExtractor()
    created = {}

    def fake_get_model(num_training_subjects, num_texture_dims, **_):
        created["num_training_subjects"] = num_training_subjects
        return extractor

    monkeypatch.setattr(train_500k, "get_DeepPrint_LocTex", fake_get_model)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_500k.py",
            "--dataset_dir", str(image_dir),
            "--output_dir", str(tmp_path / "model"),
            "--num_subjects", str(num_subjects),
            "--num_impressions", "2",
            # The test images hold no resolution
            "--assume_ppi", "500",
        ]
        + extra_args,
    )
    train_500k.main()

    fit_args = extractor.fit_args
    training_ids: IdentifierSet = fit_args["fingerprints"].ids
    labels = {fit_args["labels"].get(i) for i in training_ids}
    num_training_subjects = num_subjects - num_validation_subjects
    assert created["num_training_subjects"] == num_training_subjects
    assert labels == set(range(num_training_subjects))

    if num_validation_subjects == 0:
        assert fit_args["validation_fingerprints"] is None
        assert fit_args["validation_benchmark"] is None
    else:
        validation_ids: IdentifierSet = fit_args["validation_fingerprints"].ids
        assert validation_ids.num_subjects == num_validation_subjects
        assert {i.subject for i in validation_ids}.isdisjoint(
            {i.subject for i in training_ids}
        )
        assert fit_args["validation_benchmark"] is not None
