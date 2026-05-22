"""Target transforms for rating model. Shared by training and serving."""

import numpy as np


INVERSE_EPSILON = 0.1


def rating_inverse_transform(y):
    """Inverse target transform: 1/(6-star+0.1) spreads out low ratings."""
    return 1.0 / (6 - np.asarray(y, dtype=float) + INVERSE_EPSILON)


def rating_inverse_transform_inv(y_pred):
    """Inverse of rating_inverse_transform: maps predictions back to 1-5."""
    raw = 6 - (1.0 / np.clip(np.asarray(y_pred, dtype=float), 0.01, 10) - INVERSE_EPSILON)
    return np.clip(raw, 1, 5)


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
