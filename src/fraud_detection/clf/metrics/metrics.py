# Third Party Imports
import numpy as np


def precision_at_K(y_true: np.ndarray, y_pred_proba: np.ndarray, K: int) -> float:
    # Write checks of data types
    if not isinstance(y_true, np.ndarray):
        raise ValueError("y_true must be a numpy array")
    if not isinstance(y_pred_proba, np.ndarray):
        raise ValueError("y_pred_proba must be a numpy array")
    # Shape constraints
    if y_true.shape != y_pred_proba.shape:
        raise ValueError("y_true and y_pred_proba must have the same shape")
    if y_true.ndim != 1:
        raise ValueError("y_true must be a 1D array")
    # K Constraints
    if not isinstance(K, int):
        raise ValueError("K must be an integer")
    if K < 0 or K > len(y_pred_proba):
        raise ValueError("K must be between 0 and the length of y_pred_proba")

    # The real Calulation starts here
    top_K = np.argsort(y_pred_proba)
    y_true_top_K = y_true[top_K][-K:]
    # return the Precision
    return float(np.mean(y_true_top_K))


def recall_at_K(y_true: np.ndarray, y_pred_proba: np.ndarray, K: int) -> float:
    # Write checks of data types
    if not isinstance(y_true, np.ndarray):
        raise ValueError("y_true must be a numpy array")
    if not isinstance(y_pred_proba, np.ndarray):
        raise ValueError("y_pred_proba must be a numpy array")
    # Shape constraints
    if y_true.shape != y_pred_proba.shape:
        raise ValueError("y_true and y_pred_proba must have the same shape")
    if y_true.ndim != 1:
        raise ValueError("y_true must be a 1D array")
    # K Constraints
    if not isinstance(K, int):
        raise ValueError("K must be an integer")
    if K < 0 or K > len(y_pred_proba):
        raise ValueError("K must be between 0 and the length of y_pred_proba")

    # The real Calulation starts here
    top_K = np.argsort(y_pred_proba)
    y_true_top_K = y_true[top_K][-K:]
    # return the Recall
    return float(np.sum(y_true_top_K) / y_true.sum())


def lift_at_K(y_true: np.ndarray, y_pred_proba: np.ndarray, K: int) -> float:
    # Write checks of data types
    if not isinstance(y_true, np.ndarray):
        raise ValueError("y_true must be a numpy array")
    if not isinstance(y_pred_proba, np.ndarray):
        raise ValueError("y_pred_proba must be a numpy array")
    # Shape constraints
    if y_true.shape != y_pred_proba.shape:
        raise ValueError("y_true and y_pred_proba must have the same shape")
    if y_true.ndim != 1:
        raise ValueError("y_true must be a 1D array")
    # K Constraints
    if not isinstance(K, int):
        raise ValueError("K must be an integer")
    if K < 0 or K > len(y_pred_proba):
        raise ValueError("K must be between 0 and the length of y_pred_proba")

    # Calculate lift at top K predictions
    top_K = np.argsort(y_pred_proba)
    y_true_top_K = y_true[top_K][-K:]
    # return the lift
    return float(np.mean(y_true_top_K) / np.mean(y_true))
