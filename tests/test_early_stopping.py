"""Tests for EarlyStopping patience, mode, and reset behaviour."""

import pytest

from src.training.early_stopping import EarlyStopping


# ---------------------------------------------------------------------------
# mode: max
# ---------------------------------------------------------------------------

def test_max_mode_no_stop_while_improving() -> None:
    """Should never stop when the metric keeps increasing."""
    es = EarlyStopping(patience=3, mode="max", monitor="val_accuracy")
    values = [0.5, 0.6, 0.7, 0.8, 0.9]
    for v in values:
        assert es.step({"val_accuracy": v}) is False


def test_max_mode_stops_after_patience_exhausted() -> None:
    """Should stop exactly when bad_epochs reaches patience."""
    es = EarlyStopping(patience=3, mode="max", monitor="val_accuracy")
    es.step({"val_accuracy": 0.9})   # improvement — resets counter
    assert es.step({"val_accuracy": 0.8}) is False  # bad epoch 1
    assert es.step({"val_accuracy": 0.7}) is False  # bad epoch 2
    assert es.step({"val_accuracy": 0.6}) is True   # bad epoch 3 == patience


def test_max_mode_resets_counter_on_improvement() -> None:
    """A new best value must reset the bad-epoch counter."""
    es = EarlyStopping(patience=2, mode="max", monitor="val_accuracy")
    es.step({"val_accuracy": 0.8})   # best=0.8
    es.step({"val_accuracy": 0.7})   # bad=1
    es.step({"val_accuracy": 0.9})   # new best → bad=0
    assert es.step({"val_accuracy": 0.85}) is False  # bad=1 (< patience=2)


def test_max_mode_patience_one_stops_immediately() -> None:
    """With patience=1, one non-improvement triggers stop."""
    es = EarlyStopping(patience=1, mode="max", monitor="val_accuracy")
    es.step({"val_accuracy": 0.8})
    assert es.step({"val_accuracy": 0.7}) is True


# ---------------------------------------------------------------------------
# mode: min
# ---------------------------------------------------------------------------

def test_min_mode_no_stop_while_loss_decreasing() -> None:
    """Should not stop when loss keeps decreasing."""
    es = EarlyStopping(patience=3, mode="min", monitor="val_loss")
    values = [1.0, 0.8, 0.6, 0.4]
    for v in values:
        assert es.step({"val_loss": v}) is False


def test_min_mode_stops_after_patience_exhausted() -> None:
    """Should stop when loss stops improving for patience steps."""
    es = EarlyStopping(patience=2, mode="min", monitor="val_loss")
    es.step({"val_loss": 0.4})   # best=0.4
    assert es.step({"val_loss": 0.5}) is False  # bad=1
    assert es.step({"val_loss": 0.6}) is True   # bad=2 == patience


def test_min_mode_best_value_initialised_to_inf() -> None:
    """Initial best_value for min mode must be +inf."""
    es = EarlyStopping(patience=3, mode="min", monitor="val_loss")
    assert es.best_value == float("inf")


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------

def test_reset_clears_state() -> None:
    """reset() must restore initial best_value and zero bad_epochs."""
    es = EarlyStopping(patience=2, mode="max", monitor="val_accuracy")
    es.step({"val_accuracy": 0.9})
    es.step({"val_accuracy": 0.7})  # bad=1
    es.reset()
    assert es.bad_epochs == 0
    assert es.best_value == float("-inf")


def test_reset_allows_reuse_for_new_run() -> None:
    """After reset, the instance should behave as if freshly constructed."""
    es = EarlyStopping(patience=1, mode="max", monitor="val_accuracy")
    es.step({"val_accuracy": 0.9})
    assert es.step({"val_accuracy": 0.5}) is True  # stopped
    es.reset()
    es.step({"val_accuracy": 0.8})               # fresh best
    assert es.step({"val_accuracy": 0.7}) is True  # stops again after 1 bad


# ---------------------------------------------------------------------------
# error handling
# ---------------------------------------------------------------------------

def test_invalid_mode_raises() -> None:
    """Unsupported mode should raise ValueError at construction."""
    with pytest.raises(ValueError, match="mode"):
        EarlyStopping(patience=3, mode="sideways", monitor="val_accuracy")


def test_missing_metric_raises_key_error() -> None:
    """Missing monitored key in metrics dict must raise KeyError."""
    es = EarlyStopping(patience=3, mode="max", monitor="val_accuracy")
    with pytest.raises(KeyError, match="val_accuracy"):
        es.step({"train_loss": 0.5})


def test_patience_boundary_exact() -> None:
    """stop must be True on the exact patience-th bad epoch, not before."""
    patience = 5
    es = EarlyStopping(patience=patience, mode="max", monitor="val_accuracy")
    es.step({"val_accuracy": 1.0})  # set best
    for i in range(1, patience):
        assert es.step({"val_accuracy": 0.0}) is False, f"stopped early at bad epoch {i}"
    assert es.step({"val_accuracy": 0.0}) is True
