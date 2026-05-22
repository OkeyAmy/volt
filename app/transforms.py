"""Target transforms for rating model. Shared by training and serving."""


from __future__ import annotations

from collections.abc import Iterable

INVERSE_EPSILON = 0.1


def rating_inverse_transform(y):
    """Inverse target transform: 1/(6-star+0.1) spreads out low ratings."""
    import numpy as np

    return 1.0 / (6 - np.asarray(y, dtype=float) + INVERSE_EPSILON)


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _inverse_one(value: float) -> float:
    clipped = _clip(float(value), 0.01, 10.0)
    raw = 6 - ((1.0 / clipped) - INVERSE_EPSILON)
    return _clip(raw, 1.0, 5.0)


def rating_inverse_transform_inv(y_pred):
    """Inverse of rating_inverse_transform: maps predictions back to 1-5."""
    if isinstance(y_pred, (str, bytes)) or not isinstance(y_pred, Iterable):
        return _inverse_one(float(y_pred))
    return [_inverse_one(value) for value in y_pred]


def apply_low_rating_override(
    ridge_prediction: float,
    low_probability: float,
    low_threshold: float,
    high_threshold: float,
) -> float:
    """Tiered cap for low-rating classifier.

    prob > high_threshold → cap at 2 (very confident)
    prob > low_threshold  → cap at 3 (moderately confident)
    """
    if low_probability > high_threshold:
        return min(ridge_prediction, 2.0)
    if low_probability > low_threshold:
        return min(ridge_prediction, 3.0)
    return ridge_prediction
