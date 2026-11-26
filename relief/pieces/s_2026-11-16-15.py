"""Relief spec for "The catch": depth (mm above the sky plane) and color for each painted layer.

OVERS / SCREENS name the drawing script's over() / screen() calls in paint order (None = ignore).
DEPTH / COLOR values are constants or functions of ctx (X, Y, f, mask: the layer's coverage).
"""
import numpy as np

CX, HW = 0.45, 0.036  # ship centre line and half width, from the drawing script

OVERS = [
    "tower", "tower",            # lattice, carriage
    "far_arm",
    "ship", "flaps",
    "near_arm", "near_arm", "near_arm",  # arm, catch rail, truss
    "cloud",
    "tanks", "land",
]
SCREENS = ["flame", "glow"]
REGIONS = {"water": lambda c: (c["Y"] >= 1.075).astype(np.float32)}


def _glow(c):
    return np.exp(-(((c["X"] - CX) / 0.42) ** 2 + ((c["Y"] - 0.86) / 0.36) ** 2))


def sky_depth(c):
    return 0.0


def sky_color(c):
    g = _glow(c)[..., None]
    return np.array([0.16, 0.22, 0.48]) * (1 - g) + np.array([0.75, 0.42, 0.22]) * g


def _ship(c):
    u = np.clip((c["X"] - CX) / (HW * 1.25), -1, 1)
    return 5.0 + 3.0 * np.sqrt(1 - u * u)  # rounded hull: up to 8 mm proud


DEPTH = {
    "tower": 3.0,
    "far_arm": 4.0,
    "ship": _ship,
    "flaps": 5.5,
    "near_arm": 9.0,
    "flame": 5.0,
    "cloud": lambda c: 1.5 + 5.0 * c["mask"],  # billows roll toward the viewer
    "tanks": 2.0,
    "land": 1.5,
    "water": 0.4,
}

COLOR = {
    "tower": [0.62, 0.42, 0.30],     # rust-red steel
    "far_arm": [0.55, 0.56, 0.60],
    "ship": [0.80, 0.82, 0.88],      # stainless
    "flaps": [0.42, 0.44, 0.50],
    "near_arm": [0.60, 0.61, 0.66],
    "flame": [1.00, 0.70, 0.30],
    "glow": [1.0, 0.55, 0.22],
    "cloud": [0.92, 0.66, 0.46],
    "tanks": [0.45, 0.47, 0.55],
    "land": [0.18, 0.16, 0.15],
    "water": lambda c: np.array([0.18, 0.26, 0.48]) * (1 - _glow(c)[..., None] * 0.6)
             + np.array([0.9, 0.5, 0.25]) * _glow(c)[..., None] * 0.6,
}
