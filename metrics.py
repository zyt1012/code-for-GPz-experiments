import numpy as np


def rmse(y_true: np.ndarray, y_mean: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_mean) ** 2)))


def nlpd_gaussian(y_true: np.ndarray, y_mean: np.ndarray, y_var: np.ndarray) -> float:
    """
    Negative log predictive density under Gaussian predictive distribution.
    """
    v = np.clip(y_var, 1e-9, np.inf)
    return float(0.5 * np.mean(np.log(2.0 * np.pi * v) + (y_true - y_mean) ** 2 / v))
