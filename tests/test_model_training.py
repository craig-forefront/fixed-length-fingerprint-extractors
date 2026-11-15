import json
import math
from os.path import exists, join

import pytest
import torch

import flx.models.model_training as model_training
from flx.data.dataset import ConstantDataLoader, Dataset, Identifier, IdentifierSet
from flx.models.model_training import TrainingLog, TrainingLogEntry, train_model


def _log(tmp_path, entries: list[TrainingLogEntry]) -> TrainingLog:
    log = TrainingLog(join(tmp_path, "log.json"))
    for e in entries:
        log.add_entry(e)
    return log


def test_best_entry_has_the_lowest_equal_error_rate(tmp_path):
    log = _log(
        tmp_path,
        [
            TrainingLogEntry(1, 5.0, {}, 0.1, 0.30),
            TrainingLogEntry(2, 4.0, {}, 0.5, 0.10),
            TrainingLogEntry(3, 3.0, {}, 0.9, 0.20),
        ],
    )
    assert log.selection_metric == "validation_equal_error_rate"
    assert log.best_entry.epoch == 2
    assert log.epochs_since_best == 1


def test_best_entry_without_validation_has_the_lowest_training_loss(tmp_path):
    # The accuracy rises and the loss falls: the last epoch is the best one.
    log = _log(
        tmp_path,
        [
            TrainingLogEntry(1, 5.0, {}, 0.1),
            TrainingLogEntry(2, 4.0, {}, 0.5),
            TrainingLogEntry(3, 3.0, {}, 0.9),
        ],
    )
    assert log.selection_metric == "training_loss"
    assert log.best_entry.epoch == 3
    assert log.epochs_since_best == 0
    assert all(e.validation_equal_error_rate is None for e in log._entries)


def test_best_entry_is_never_an_epoch_with_a_loss_that_is_not_a_number(tmp_path):
    log = _log(
        tmp_path,
        [
            TrainingLogEntry(1, 5.0, {}, 0.1),
            TrainingLogEntry(2, math.nan, {}, 0.0),
        ],
    )
    assert log.best_entry.epoch == 1
    log = _log(
        join(tmp_path),
        [TrainingLogEntry(3, 4.0, {}, 0.2, math.nan)],
    )
    assert log.selection_metric == "validation_equal_error_rate"
    # No epoch has a usable error rate: the first one is kept
    assert log.best_entry.epoch == 1


def test_epochs_without_validation_lose_against_validated_epochs(tmp_path):
    log = _log(
        tmp_path,
        [
            TrainingLogEntry(1, 0.1, {}, 0.99),
            TrainingLogEntry(2, 4.0, {}, 0.5, 0.25),
        ],
    )
    assert log.best_entry.epoch == 2


def test_empty_log_has_no_best_entry(tmp_path):
    log = _log(tmp_path, [])
    assert log.best_entry is None
    assert log.epochs_since_best == 0


def test_log_survives_saving_and_loading(tmp_path):
    _log(
        tmp_path,
        [TrainingLogEntry(1, 5.0, {}, 0.1), TrainingLogEntry(2, 4.0, {}, 0.5, 0.2)],
    )
    with open(join(tmp_path, "log.json")) as file:
        stored = json.load(file)["entries"]
    assert stored[0]["validation_equal_error_rate"] is None
    log = TrainingLog(join(tmp_path, "log.json"))
    assert len(log) == 2
    assert log.best_entry.epoch == 2


def test_old_log_with_accuracy_stored_as_error_rate_is_read_as_no_validation(tmp_path):
    # Logs written before the fix hold the training accuracy in the error rate
    # field when there was no validation set.
    old = {
        "entries": [
            {
                "epoch": i + 1,
                "training_loss": 5.0 - i,
                "loss_statistics": {},
                "training_accuracy": acc,
                "validation_equal_error_rate": acc,
            }
            for i, acc in enumerate([0.1, 0.5, 0.9])
        ]
    }
    with open(join(tmp_path, "log.json"), "w") as file:
        json.dump(old, file)
    log = TrainingLog(join(tmp_path, "log.json"))
    assert log.selection_metric == "training_loss"
    assert log.best_entry.epoch == 3


# --- train_model with scripted epochs -----------------------------------------


class _NoLoss(torch.nn.Module):
    pass


def _run_scripted_training(
    monkeypatch, out_dir, epochs: list[tuple], validation: bool, patience: int = 0
) -> tuple[list[int], int]:
    """
    Runs train_model with the given results per epoch instead of real training:
    (training loss, training accuracy, validation equal error rate).

    Returns the epochs that were trained and the epoch stored as best model.
    """
    trained = []

    def fake_train(*args, **kwargs):
        trained.append(len(trained))
        loss, accuracy, _ = epochs[trained[-1]]
        return loss, {}, accuracy

    def fake_validate(*args, **kwargs):
        return epochs[trained[-1]][2]

    def fake_save(path, model, loss, optimizer):
        with open(path, "w") as file:
            file.write(str(len(trained)))

    monkeypatch.setattr(model_training, "_train", fake_train)
    monkeypatch.setattr(model_training, "_validate", fake_validate)
    monkeypatch.setattr(model_training, "save_model_parameters", fake_save)
    monkeypatch.setattr(model_training, "load_model_parameters", lambda *a: None)

    ids = IdentifierSet([Identifier(0, 0), Identifier(1, 0)])
    data = Dataset(ConstantDataLoader(0), ids)
    train_model(
        fingerprints=data,
        minutia_maps=data,
        labels=data,
        validation_fingerprints=data if validation else None,
        validation_benchmark=object() if validation else None,
        model=torch.nn.Linear(1, 1),
        loss=_NoLoss(),
        num_epochs=len(epochs),
        out_dir=str(out_dir),
        patience=patience,
        use_amp=False,
    )
    with open(join(out_dir, "best_model.pyt")) as file:
        best = int(file.read())
    return [t + 1 for t in trained], best


def test_training_without_validation_keeps_the_epoch_with_the_lowest_loss(
    monkeypatch, tmp_path
):
    # Accuracy rises from epoch to epoch. The old code stored it as error rate
    # and kept the epoch with the lowest one, which is the first.
    epochs = [(5.0, 0.1, None), (4.0, 0.5, None), (3.0, 0.9, None)]
    trained, best = _run_scripted_training(monkeypatch, tmp_path, epochs, False)
    assert trained == [1, 2, 3]
    assert best == 3
    with open(join(tmp_path, "log.json")) as file:
        entries = json.load(file)["entries"]
    assert [e["validation_equal_error_rate"] for e in entries] == [None] * 3
    assert [e["training_accuracy"] for e in entries] == [0.1, 0.5, 0.9]


def test_training_without_validation_does_not_stop_early(monkeypatch, tmp_path):
    epochs = [(3.0, 0.9, None), (4.0, 0.5, None), (5.0, 0.1, None)]
    trained, best = _run_scripted_training(
        monkeypatch, tmp_path, epochs, False, patience=1
    )
    assert trained == [1, 2, 3]
    assert best == 1


def test_training_with_validation_keeps_the_epoch_with_the_lowest_error_rate(
    monkeypatch, tmp_path
):
    epochs = [(5.0, 0.1, 0.30), (4.0, 0.5, 0.10), (3.0, 0.9, 0.20), (2.0, 1.0, 0.15)]
    trained, best = _run_scripted_training(monkeypatch, tmp_path, epochs, True)
    assert trained == [1, 2, 3, 4]
    assert best == 2


def test_training_stops_after_patience_epochs_without_a_lower_error_rate(
    monkeypatch, tmp_path
):
    epochs = [(5.0, 0.1, 0.30), (4.0, 0.5, 0.10), (3.0, 0.9, 0.20), (2.0, 1.0, 0.10)]
    epochs += [(1.0, 1.0, 0.05)]
    trained, best = _run_scripted_training(
        monkeypatch, tmp_path, epochs, True, patience=2
    )
    # Epoch 4 only equals epoch 2, which is not an improvement
    assert trained == [1, 2, 3, 4]
    assert best == 2


def test_validation_set_and_benchmark_must_come_together(tmp_path):
    ids = IdentifierSet([Identifier(0, 0)])
    data = Dataset(ConstantDataLoader(0), ids)
    with pytest.raises(ValueError):
        train_model(
            fingerprints=data,
            minutia_maps=data,
            labels=data,
            validation_fingerprints=data,
            validation_benchmark=None,
            model=torch.nn.Linear(1, 1),
            loss=_NoLoss(),
            num_epochs=1,
            out_dir=str(tmp_path),
        )
    assert not exists(join(tmp_path, "best_model.pyt"))
