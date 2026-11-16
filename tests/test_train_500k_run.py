"""
scripts/train_500k.py as a whole: the file layout it reads, continuing a
training and starting again, the batch size, and loading the trained model.
"""

import importlib.util
import json
import sys
from os.path import abspath, dirname, exists, join

import cv2
import numpy as np
import pytest

from flx.data.dataset import Dataset, Identifier, IdentifierSet
from flx.models import torch_helpers
from flx.models.model_training import TrainingLog, TrainingLogEntry
from flx.preparation.sensor_normalization import SensorNormalizer

SCRIPTS_DIR = join(dirname(dirname(abspath(__file__))), "scripts")


@pytest.fixture(scope="module")
def train_500k():
    spec = importlib.util.spec_from_file_location(
        "train_500k", join(SCRIPTS_DIR, "train_500k.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def default_batch_size():
    yield
    torch_helpers.set_training_batch_size(torch_helpers.DEFAULT_TRAINING_BATCH_SIZE)


class _FakeExtractor:
    """Its fit writes model.pyt and one log entry per epoch, like train_model."""

    def __init__(self):
        self.fits = []
        self.model = "the model"

    def fit(self, out_dir, num_epochs, **kwargs):
        self.fits.append(dict(kwargs, num_epochs=num_epochs))
        log = TrainingLog(join(out_dir, "log.json"))
        for epoch in range(len(log) + 1, num_epochs + 1):
            log.add_entry(TrainingLogEntry(epoch, 5.0 - epoch, {}, 0.1, None, 1e-4))
            for name in ("model.pyt", "best_model.pyt"):
                with open(join(out_dir, name), "w") as file:
                    file.write(str(epoch))


def _write_images(image_dir, subjects=range(1, 5), impressions=(1, 2), subdir="", size=8):
    directory = image_dir / subdir
    directory.mkdir(parents=True, exist_ok=True)
    for subject in subjects:
        for impression in impressions:
            cv2.imwrite(
                str(directory / f"{subject}_{impression}.png"),
                np.zeros((size, size), dtype=np.uint8),
            )


def _write_minutiae(minutiae_dir, subjects=range(1, 5), impressions=(1, 2)):
    minutiae_dir.mkdir(parents=True, exist_ok=True)
    for subject in subjects:
        for impression in impressions:
            (minutiae_dir / f"{subject}_{impression}.mnt").write_text("1\n8 8\n4 4 0.5\n")


def _run(train_500k, tmp_path, monkeypatch, *options, output="model", fake=None):
    """Runs main() with a fake model on tmp_path/images. Returns the fake."""
    fake = fake or _FakeExtractor()
    for variant in ["LocTexMinu", "LocTex", "TexMinu", "Tex"]:
        monkeypatch.setattr(
            train_500k, "get_DeepPrint_" + variant, lambda *a, **k: fake
        )
    argv = [
        "train_500k.py",
        "--dataset_dir", str(tmp_path / "images"),
        "--output_dir", str(tmp_path / output),
        "--num_subjects", "4",
        "--num_impressions", "2",
        "--assume_ppi", "500",  # the test images hold no resolution
        "--validation_split", "0",
        "--num_epochs", "2",
    ]
    monkeypatch.setattr(sys, "argv", argv + [str(option) for option in options])
    train_500k.main()
    return fake


def _stops(train_500k, tmp_path, monkeypatch, capsys, *options, **kwargs) -> str:
    """Runs main(), which must stop with exit status 2. Returns its message."""
    capsys.readouterr()
    with pytest.raises(SystemExit) as stop:
        _run(train_500k, tmp_path, monkeypatch, *options, **kwargs)
    assert stop.value.code == 2
    return capsys.readouterr().err


# --- The file layout ----------------------------------------------------------


def test_missing_files_are_named_with_the_options_that_ask_for_them(
    train_500k, tmp_path, monkeypatch, capsys
):
    # 4 subjects with 2 impressions, but the options ask for 5 with 3
    _write_images(tmp_path / "images")
    message = _stops(
        train_500k, tmp_path, monkeypatch, capsys,
        "--num_subjects", 5, "--num_impressions", 3,
    )
    assert "7 of the 15 files that the training needs are missing" in message
    assert "1_3.png, 2_3.png, 3_3.png" in message
    assert "--num_subjects 5 and --num_impressions 3" in message
    assert "8 .png files of 4 subjects were found" in message
    assert "impression numbers 1 to 2" in message
    assert not exists(tmp_path / "model" / "model.pyt")


def test_subject_with_fewer_impressions_is_named(
    train_500k, tmp_path, monkeypatch, capsys
):
    _write_images(tmp_path / "images")
    (tmp_path / "images" / "3_2.png").unlink()
    message = _stops(train_500k, tmp_path, monkeypatch, capsys)
    assert "1 of the 8 files" in message and "3_2.png" in message


def test_missing_minutiae_files_are_named(train_500k, tmp_path, monkeypatch, capsys):
    _write_images(tmp_path / "images")
    _write_minutiae(tmp_path / "minutiae", subjects=range(1, 4))
    message = _stops(
        train_500k, tmp_path, monkeypatch, capsys,
        "--minutiae_dir", tmp_path / "minutiae",
    )
    assert "2 of the 8 files" in message and "4_1.mnt, 4_2.mnt" in message
    assert "--minutiae_dir" in message and "extract_minutiae.py" in message


def test_sensor_directories_are_read_and_files_that_are_not_used_are_counted(
    train_500k, tmp_path, monkeypatch, capsys
):
    _write_images(tmp_path / "images", subjects=(1, 2), subdir="sensor_a")
    _write_images(tmp_path / "images", subjects=(3, 4, 5), subdir="sensor_b")
    _write_images(tmp_path / "images", subjects=(1,), impressions=(3,), subdir="sensor_b")
    fake = _run(train_500k, tmp_path, monkeypatch)
    assert len(fake.fits[0]["fingerprints"]) == 8
    output = capsys.readouterr().out
    assert "3 image files below" in output and "are not used" in output


def test_same_file_name_in_two_directories_is_an_error(
    train_500k, tmp_path, monkeypatch, capsys
):
    # Before, one of the two files was dropped without a message
    _write_images(tmp_path / "images", subdir="sensor_a")
    _write_images(tmp_path / "images", subjects=(1,), impressions=(1,), subdir="sensor_b")
    message = _stops(train_500k, tmp_path, monkeypatch, capsys)
    assert "have the same identifier" in message
    assert "sensor_a/1_1.png" in message and "sensor_b/1_1.png" in message


@pytest.mark.parametrize("name", ["subject_001_impression_01.png", "a_1.png", "0_1.png"])
def test_file_name_that_is_not_subject_and_impression_is_an_error(
    train_500k, tmp_path, monkeypatch, capsys, name
):
    _write_images(tmp_path / "images")
    cv2.imwrite(str(tmp_path / "images" / name), np.zeros((8, 8), dtype=np.uint8))
    message = _stops(train_500k, tmp_path, monkeypatch, capsys)
    assert name in message and "<subject>_<impression>" in message


def test_resolution_sources_of_the_training_images_are_printed(
    train_500k, tmp_path, monkeypatch, capsys
):
    from PIL import Image

    _write_images(tmp_path / "images")
    # An image program stamped 300 ppi into three files
    for name in ("1_1.png", "1_2.png", "2_1.png"):
        Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(
            str(tmp_path / "images" / name), dpi=(300, 300)
        )
    _run(train_500k, tmp_path, monkeypatch)
    lines = [" ".join(line.split()) for line in capsys.readouterr().out.splitlines()]
    assert "Resolution of 8 training images spread over the set:" in lines
    assert "5 500 ppi from --assume_ppi" in lines
    assert "3 300 ppi from metadata of the file (resampled to 500 ppi)" in lines
    assert any("does not override" in line for line in lines)


def test_resolution_sample_is_spread_over_the_list(train_500k):
    class Normalizer:
        def resolve_ppi(self, path):
            if path == "bad":
                raise ValueError("unknown")
            return (500.0, "argument") if int(path) < 5000 else (1000.0, "manifest")

    paths = [str(i) for i in range(10000)] + ["bad"]
    counts, looked_at = train_500k.sample_resolution_sources(Normalizer(), paths, 100)
    assert looked_at == 100
    assert counts == {"argument, 500 ppi": 50, "manifest, 1000 ppi": 50}
    counts, looked_at = train_500k.sample_resolution_sources(Normalizer(), ["1", "bad"])
    assert counts == {"argument, 500 ppi": 1, "unknown": 1}


# --- Continuing a training and starting again ---------------------------------


def test_same_command_continues_and_a_finished_training_says_so(
    train_500k, tmp_path, monkeypatch, capsys
):
    _write_images(tmp_path / "images")
    _run(train_500k, tmp_path, monkeypatch)
    output = capsys.readouterr().out
    assert "Training complete! 2 epoch(s) trained in this run, 2 in the log." in output
    with open(tmp_path / "model" / "training_settings.json") as file:
        settings = json.load(file)
    assert settings["variant"] == "LocTex"
    assert settings["num_training_subjects"] == 4
    assert settings["batch_size"] == 64
    assert settings["preparation"]["assume_ppi"] == 500

    # The same command with another learning rate: nothing is left to train,
    # and the script must not claim that it trained
    _run(train_500k, tmp_path, monkeypatch, "--learning_rate", 0.00003)
    output = capsys.readouterr().out
    assert "Nothing was trained in this run" in output
    assert "Training complete" not in output
    assert "learning_rate 0.0001 -> 3e-05" in output
    assert "use a new --output_dir" in output

    # More epochs continue the stored training
    _run(train_500k, tmp_path, monkeypatch, "--learning_rate", 0.00003, "--num_epochs", 3)
    output = capsys.readouterr().out
    assert "Training complete! 1 epoch(s) trained in this run, 3 in the log." in output
    assert "num_epochs 2 -> 3" in output
    assert "learning_rate" not in [l.split(":")[1].split()[0] for l in output.splitlines()
                                   if l.startswith("Note: Changed since")]


@pytest.mark.parametrize(
    "options, changed",
    [
        (["--localization", "off"], "variant"),
        (["--embedding_dim", 128], "embedding_dim"),
        (["--batch_size", 32], "batch_size"),
        (["--num_subjects", 3], "num_subjects"),
        (["--canvas_size", 512], "canvas_size"),
        (["--ridges", "light"], "preparation"),
        (["--assume_ppi", 1000], "preparation"),
    ],
)
def test_continuing_with_settings_that_cannot_change_is_refused(
    train_500k, tmp_path, monkeypatch, capsys, options, changed
):
    _write_images(tmp_path / "images")
    first = _run(train_500k, tmp_path, monkeypatch)
    again = _FakeExtractor()
    message = _stops(train_500k, tmp_path, monkeypatch, capsys, *options, fake=again)
    assert f"  {changed}: stored " in message
    assert "use a new --output_dir" in message
    assert "training_settings.json" in message
    assert again.fits == [] and len(first.fits) == 1
    # The stored settings are still those of the training
    with open(tmp_path / "model" / "training_settings.json") as file:
        assert json.load(file)["variant"] == "LocTex"

    # In a new output directory the same options train
    _run(train_500k, tmp_path, monkeypatch, *options, output="model_2")
    assert exists(tmp_path / "model_2" / "model.pyt")


def test_refusal_can_be_overridden(train_500k, tmp_path, monkeypatch, capsys):
    _write_images(tmp_path / "images")
    _run(train_500k, tmp_path, monkeypatch)
    capsys.readouterr()
    _run(
        train_500k, tmp_path, monkeypatch,
        "--batch_size", 32, "--num_epochs", 3, "--allow_changed_settings",
    )
    output = capsys.readouterr().out
    assert "Continuing although batch_size changed" in output
    assert "1 epoch(s) trained in this run" in output


def test_training_of_an_earlier_version_is_compared_by_its_preparation(
    train_500k, tmp_path, monkeypatch, capsys
):
    _write_images(tmp_path / "images")
    _run(train_500k, tmp_path, monkeypatch)
    (tmp_path / "model" / "training_settings.json").unlink()
    capsys.readouterr()
    # data_preparation.json is still there: another preparation is refused
    message = _stops(train_500k, tmp_path, monkeypatch, capsys, "--ridges", "light")
    assert "  preparation: stored " in message
    _run(train_500k, tmp_path, monkeypatch, "--num_epochs", 3)
    output = capsys.readouterr().out
    assert "was started before training_settings.json existed" in output
    assert exists(tmp_path / "model" / "training_settings.json")


def test_cosine_schedule_with_more_epochs_is_pointed_out(train_500k, tmp_path):
    def settings(**changes):
        base = {key: 1 for key in train_500k.FIXED_SETTINGS}
        base.update({key: 1 for key in train_500k.CHANGEABLE_SETTINGS})
        base.update(lr_schedule="cosine", **changes)
        return base

    assert train_500k.check_training_settings(str(tmp_path), settings()) == []
    (tmp_path / "model.pyt").write_text("")
    assert train_500k.check_training_settings(str(tmp_path), settings()) == []
    notes = train_500k.check_training_settings(str(tmp_path), settings(num_epochs=5))
    assert len(notes) == 2
    assert "num_epochs 1 -> 5" in notes[0]
    assert "does not continue where the last one ended" in notes[1]
    with pytest.raises(train_500k.SettingsChangedError, match="variant"):
        train_500k.check_training_settings(str(tmp_path), settings(variant=2))


# --- Batch size and validation limit ------------------------------------------


def test_batch_size_option_sets_the_batch_of_the_training(
    train_500k, tmp_path, monkeypatch, capsys
):
    _write_images(tmp_path / "images")
    assert torch_helpers.get_dataloader_args(train=True)["batch_size"] == 64
    _run(train_500k, tmp_path, monkeypatch, "--batch_size", 24)
    assert torch_helpers.get_dataloader_args(train=True)["batch_size"] == 24
    assert torch_helpers.get_dataloader_args(train=False)["batch_size"] == 48
    assert "Batch size:   24 images per step and GPU" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        _run(train_500k, tmp_path, monkeypatch, "--batch_size", 0, output="other")


def test_validation_subjects_are_limited_by_default(
    train_500k, tmp_path, monkeypatch, capsys
):
    seen = []
    real = train_500k.create_training_and_validation_ids

    def record(num_subjects, num_impressions, split, seed, limit):
        seen.append(limit)
        return real(num_subjects, num_impressions, split, seed, limit)

    monkeypatch.setattr(train_500k, "create_training_and_validation_ids", record)
    _write_images(tmp_path / "images")
    _run(train_500k, tmp_path, monkeypatch)
    _run(train_500k, tmp_path, monkeypatch, "--max_validation_subjects", 0, output="b")
    assert seen == [5000, 0]
    assert train_500k.DEFAULT_MAX_VALIDATION_SUBJECTS == 5000
    # The limit keeps the other subjects in the training set
    train, validation = train_500k.split_subjects(200_000, 0.1, 42, 5000)
    assert (len(train), len(validation)) == (195_000, 5000)

    # A limit that takes effect is reported
    capsys.readouterr()
    _write_images(tmp_path / "images", subjects=range(5, 41))
    _run(
        train_500k, tmp_path, monkeypatch, "--num_subjects", 40,
        "--validation_split", 0.5, "--max_validation_subjects", 3, output="c",
    )
    output = capsys.readouterr().out
    assert "would hold out 20 subjects; --max_validation_subjects limits this to 3" in output


# --- Using the trained model ---------------------------------------------------


def test_trained_model_is_loaded_with_the_settings_of_its_training(
    train_500k, tmp_path, monkeypatch
):
    _write_images(tmp_path / "images", subjects=range(1, 21), size=200)
    _write_minutiae(tmp_path / "minutiae", subjects=range(1, 21))
    manifest = tmp_path / "resolutions.csv"
    manifest.write_text("path,ppi,ridges\n.,1000,light\n")
    _run(
        train_500k, tmp_path, monkeypatch,
        "--num_subjects", 20, "--validation_split", 0.2,
        "--minutiae_dir", tmp_path / "minutiae", "--localization", "off",
        "--embedding_dim", 96, "--canvas_size", 512, "--ppi_manifest", manifest,
    )

    built, loaded = [], []
    fake = _FakeExtractor()

    def constructor(**kwargs):
        built.append(kwargs)
        return fake

    monkeypatch.setattr(train_500k, "get_DeepPrint_TexMinu", constructor)
    monkeypatch.setattr(
        train_500k.torch_helpers, "load_model_parameters", lambda *a: loaded.append(a)
    )
    extractor, image_loader = train_500k.load_trained_model(
        str(tmp_path / "model"), str(tmp_path / "images")
    )
    assert extractor is fake
    # One class per TRAINING subject: 20 subjects minus the 4 held out
    assert built == [
        {"num_training_subjects": 16, "num_dims": 96, "loss_weights": None}
    ]
    assert loaded == [(str(tmp_path / "model" / "best_model.pyt"), "the model", None, None)]
    # The images are prepared as in the training: 1000 ppi halved, canvas 512
    dataset = Dataset(image_loader, IdentifierSet([Identifier(0, 0)]))
    assert len(image_loader.ids) == 40
    assert tuple(dataset[0].shape[-2:]) == (299, 299)
    normalizer = image_loader._images._normalizer
    expected = SensorNormalizer(
        root_dir=str(tmp_path / "images"), manifest=str(manifest), assume_ppi=500
    ).settings()
    assert normalizer.settings() == expected

    with pytest.raises(FileNotFoundError, match="training_settings.json"):
        train_500k.load_trained_model(str(tmp_path / "nothing"), str(tmp_path / "images"))
