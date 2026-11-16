import json
import math
from os.path import exists, join

import pytest
import torch

import flx.models.model_training as model_training
from flx.data.dataset import (
    ConstantDataLoader,
    DataLoader,
    Dataset,
    Identifier,
    IdentifierSet,
)
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


def test_an_epoch_that_is_not_a_number_loses_also_when_it_comes_first(tmp_path):
    # min() keeps the first of two values that do not compare, so the order
    # of the entries must not decide
    log = _log(
        join(tmp_path),
        [TrainingLogEntry(1, math.nan, {}, 0.0), TrainingLogEntry(2, 5.0, {}, 0.1)],
    )
    assert log.best_entry.epoch == 2
    log = _log(
        join(tmp_path),
        [
            TrainingLogEntry(3, 4.0, {}, 0.2, math.nan),
            TrainingLogEntry(4, 4.0, {}, 0.2, 0.4),
        ],
    )
    assert log.best_entry.epoch == 4


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


def test_equal_error_rate_and_accuracy_in_a_new_log_are_a_real_result(tmp_path):
    # A training accuracy of exactly 0 is normal in the first epoch with many
    # classes, and a small validation set can give an error rate of exactly 0.
    # Logs that record the learning rate never hold the accuracy in the error
    # rate field, so the value must be kept.
    _log(
        tmp_path,
        [
            TrainingLogEntry(1, 5.0, {}, 0.0, 0.0, 1e-4),
            TrainingLogEntry(2, 4.0, {}, 0.5, 0.3, 1e-4),
        ],
    )
    log = TrainingLog(join(tmp_path, "log.json"))
    assert log.best_entry.epoch == 1
    assert log._entries[0].validation_equal_error_rate == 0.0


# --- train_model with scripted epochs -----------------------------------------


class _NoLoss(torch.nn.Module):
    pass


def _run_scripted_training(
    monkeypatch,
    out_dir,
    epochs: list[tuple],
    validation: bool,
    patience: int = 0,
    already_trained: int = 0,
) -> tuple[list[int], int]:
    """
    Runs train_model with the given results per epoch instead of real training:
    (training loss, training accuracy, validation equal error rate).

    Returns the epochs that were trained and the epoch stored as best model.
    """
    trained = list(range(already_trained))

    def fake_train(*args, **kwargs):
        trained.append(len(trained))
        loss, accuracy, _ = epochs[trained[-1]]
        return loss, {}, accuracy

    def fake_validate(*args, **kwargs):
        result = epochs[trained[-1]][2]
        if isinstance(result, BaseException):
            raise result
        return result

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
    return [t + 1 for t in trained[already_trained:]], best


def test_run_stopped_during_validation_resumes_with_the_next_epoch(
    monkeypatch, tmp_path
):
    # The run is stopped while epoch 2 is validated. model.pyt already holds
    # epoch 2, so the log must hold it too: the next run must not train
    # epoch 2 again on parameters that are already one epoch further.
    stopped = [(5.0, 0.1, 0.30), (4.0, 0.5, KeyboardInterrupt())]
    with pytest.raises(KeyboardInterrupt):
        _run_scripted_training(monkeypatch, tmp_path, stopped, True)
    with open(join(tmp_path, "log.json")) as file:
        entries = json.load(file)["entries"]
    assert [e["epoch"] for e in entries] == [1, 2]
    assert entries[1]["validation_equal_error_rate"] is None
    with open(join(tmp_path, "model.pyt")) as file:
        assert file.read() == "2"

    # The next run first validates epoch 2 (the best so far), then trains 3
    epochs = [(5.0, 0.1, 0.30), (4.0, 0.5, 0.10), (3.0, 0.9, 0.20)]
    trained, best = _run_scripted_training(
        monkeypatch, tmp_path, epochs, True, already_trained=2
    )
    assert trained == [3]
    assert best == 2
    with open(join(tmp_path, "log.json")) as file:
        entries = json.load(file)["entries"]
    assert [e["validation_equal_error_rate"] for e in entries] == [0.30, 0.10, 0.20]


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


def test_training_that_stopped_early_is_not_continued_by_the_same_command(
    monkeypatch, tmp_path, capsys
):
    epochs = [(5.0, 0.1, 0.10), (4.0, 0.5, 0.20), (3.0, 0.9, 0.05)]
    trained, best = _run_scripted_training(
        monkeypatch, tmp_path, epochs, True, patience=1
    )
    assert trained == [1, 2]
    # The same call again: epoch 3 would be the best, but the training ended
    trained, best = _run_scripted_training(
        monkeypatch, tmp_path, epochs, True, patience=1, already_trained=2
    )
    assert trained == []
    assert best == 1
    assert "nothing more is trained" in capsys.readouterr().out
    # With more patience the same directory is continued
    trained, best = _run_scripted_training(
        monkeypatch, tmp_path, epochs, True, patience=2, already_trained=2
    )
    assert trained == [3]
    assert best == 3


def test_finished_training_says_that_nothing_is_trained(monkeypatch, tmp_path, capsys):
    epochs = [(5.0, 0.1, None), (4.0, 0.5, None)]
    _run_scripted_training(monkeypatch, tmp_path, epochs, False)
    capsys.readouterr()
    trained, _ = _run_scripted_training(
        monkeypatch, tmp_path, epochs, False, already_trained=2
    )
    assert trained == []
    assert "already has 2 of 2 epochs" in capsys.readouterr().out


def test_validation_set_and_benchmark_must_come_together(tmp_path):
    ids = IdentifierSet([Identifier(0, 0)])
    data = Dataset(ConstantDataLoader(0), ids)
    with pytest.raises(ValueError, match="must be given together"):
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
    # Refused at once: nothing was trained or written
    assert not exists(join(tmp_path, "best_model.pyt"))
    assert not exists(join(tmp_path, "model.pyt"))
    assert not exists(join(tmp_path, "log.json"))


# --- _train -------------------------------------------------------------------


class _ConstantLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, output, labels, minutia_maps, minutia_map_weights):
        return 2.0 + 0.0 * self.weight.sum() + 0.0 * output.texture_logits.sum()

    def reset_recorded_loss(self):
        pass

    def get_recorded_loss(self):
        return {}


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 8)

    def forward(self, images):
        from flx.models.deep_print_arch import DeepPrintTrainingOutput

        return DeepPrintTrainingOutput(texture_logits=self.linear(images))


def test_epoch_loss_is_the_mean_over_the_samples_of_the_sampler():
    # In multi-GPU training every process gets a sampler with a part of the
    # training set. The loss of each sample is 2, so the mean must be 2.
    ids = IdentifierSet([Identifier(s, 0) for s in range(8)])
    train_set = Dataset.zip(
        Dataset(ConstantDataLoader(torch.zeros(1)), ids),
        Dataset(ConstantDataLoader((torch.zeros(0), 0.0)), ids),
        Dataset(ConstantDataLoader(0), ids),
    )
    model = _TinyModel()
    sampler = torch.utils.data.SubsetRandomSampler([0, 1, 2, 3])
    mean_loss, _, _ = model_training._train(
        model,
        _ConstantLoss(),
        torch.optim.SGD(model.parameters(), lr=0.1),
        train_set,
        sampler,
        use_amp=False,
    )
    assert mean_loss == pytest.approx(2.0)


class _PerSubjectLoader(DataLoader):
    """Returns the value that the function gives for the subject."""

    def __init__(self, function):
        self._function = function

    def get(self, identifier):
        return self._function(identifier.subject)


class _ModelThatNamesTheClassInTheImage(torch.nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, images):
        from flx.models.deep_print_arch import DeepPrintTrainingOutput

        named = images.flatten().long()
        logits = torch.nn.functional.one_hot(named, self.num_classes).float()
        return DeepPrintTrainingOutput(texture_logits=logits + 0.0 * self.weight)


def test_accuracy_is_the_share_of_samples_named_correctly_also_with_many_classes():
    # 200,000 classes: the metric that was used before needs the square of
    # the number of classes in memory (320 GB here) and failed from about
    # 100,000 classes. The model names the right subject for 6 of 8 samples.
    num_classes = 200_000
    subjects = [0, 1, 2, 3, 50_000, 100_000, 150_000, 199_999]
    wrong = {2: 7, 150_000: 3}  # subject -> the class that the model names
    ids = IdentifierSet([Identifier(s, 0) for s in subjects])
    train_set = Dataset.zip(
        Dataset(_PerSubjectLoader(lambda s: torch.tensor([float(wrong.get(s, s))])), ids),
        Dataset(ConstantDataLoader((torch.zeros(0), 0.0)), ids),
        Dataset(_PerSubjectLoader(lambda s: s), ids),
    )
    model = _ModelThatNamesTheClassInTheImage(num_classes)
    _, _, accuracy = model_training._train(
        model,
        _ConstantLoss(),
        torch.optim.SGD(model.parameters(), lr=0.1),
        train_set,
        None,
        use_amp=False,
    )
    assert accuracy == pytest.approx(6 / 8)
