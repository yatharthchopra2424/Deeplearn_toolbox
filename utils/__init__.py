"""
utils/__init__.py
==================
Shared math helpers used across multiple model modules.
Import with: from utils import sigmoid, relu, softmax, ...
"""

import numpy as np


# ── Activations ───────────────────────────────────────────────────────────────
def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid: σ(z) = 1 / (1 + e^{-z})"""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_deriv(a: np.ndarray) -> np.ndarray:
    """Derivative of sigmoid given its OUTPUT a: σ'= a(1-a)"""
    return a * (1.0 - a)

def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)

def relu_deriv(a: np.ndarray) -> np.ndarray:
    return (a > 0).astype(float)

def tanh_act(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)

def tanh_deriv(a: np.ndarray) -> np.ndarray:
    return 1.0 - a ** 2

def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - np.max(z))
    return e / (e.sum() + 1e-12)

def leaky_relu(z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    return np.where(z > 0, z, alpha * z)


# ── Loss functions ────────────────────────────────────────────────────────────
def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))

def binary_cross_entropy(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          eps: float = 1e-12) -> float:
    yp = np.clip(y_pred, eps, 1 - eps)
    return float(-np.mean(
        y_true * np.log(yp) + (1 - y_true) * np.log(1 - yp)))

def categorical_cross_entropy(y_true: np.ndarray,
                               y_pred: np.ndarray,
                               eps: float = 1e-12) -> float:
    return float(-np.sum(y_true * np.log(np.clip(y_pred, eps, 1))))


# ── Weight initialisers ───────────────────────────────────────────────────────
def xavier_uniform(fan_in: int, fan_out: int) -> np.ndarray:
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, (fan_out, fan_in))

def he_normal(fan_in: int, fan_out: int) -> np.ndarray:
    std = np.sqrt(2.0 / fan_in)
    return np.random.randn(fan_out, fan_in) * std


# ── Misc ──────────────────────────────────────────────────────────────────────
def one_hot_encode(idx: int, size: int) -> np.ndarray:
    v = np.zeros(size)
    v[idx] = 1.0
    return v

def normalize(X: np.ndarray) -> np.ndarray:
    """Min-max normalise each feature column to [0, 1]."""
    mins  = X.min(axis=0)
    maxes = X.max(axis=0)
    denom = np.where(maxes - mins == 0, 1, maxes - mins)
    return (X - mins) / denom

def clip_gradients(grads: list, clip_val: float = 5.0) -> list:
    return [np.clip(g, -clip_val, clip_val) for g in grads]
