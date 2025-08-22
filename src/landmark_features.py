from __future__ import annotations
import math
from typing import List, Tuple
import numpy as np

KEY_IDXS = [0,1,5,9,13,17,4,8,12,16,20]  # wrist, MCPs & fingertips

def _rotate2d(xy: np.ndarray, angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    R = np.array([[c, -s],[s, c]], dtype=np.float32)
    return (R @ xy.T).T

def canonicalize_landmarks(
    lms_xy_norm: np.ndarray, handedness: str | None
) -> tuple[np.ndarray, float]:
    pts = lms_xy_norm.copy().astype(np.float32)

    # Miroir si main gauche (uniformisation)
    if handedness and handedness.lower().startswith("left"):
        pts[:,0] = 1.0 - pts[:,0]

    # Origine au poignet
    wrist = pts[0].copy()
    pts -= wrist

    # Mise à l’échelle par ||poignet -> MCP majeur||
    v = pts[9].copy()
    scale = np.linalg.norm(v) + 1e-6
    pts /= scale

    # Rotation: vecteur poignet->MCP majeur aligné vertical
    angle = math.atan2(v[1], v[0])
    target = math.pi/2
    rot = target - angle
    pts = _rotate2d(pts, rot)
    return pts, scale

def pairwise_dists(points: np.ndarray) -> np.ndarray:
    sel = points[KEY_IDXS]
    n = sel.shape[0]
    feats = []
    for i in range(n):
        for j in range(i+1, n):
            feats.append(np.linalg.norm(sel[i] - sel[j]))
    return np.array(feats, dtype=np.float32)  # 55

def finger_curls(points: np.ndarray) -> np.ndarray:
    pairs = [(4,1),(8,5),(12,9),(16,13),(20,17)]
    vals = [np.linalg.norm(points[tip] - points[mcp]) for tip, mcp in pairs]
    return np.array(vals, dtype=np.float32)  # 5

def build_feature_vector(lms_xy_norm: np.ndarray, handedness: str | None) -> np.ndarray:
    pts_canon, _ = canonicalize_landmarks(lms_xy_norm, handedness)
    xy_flat = pts_canon.flatten()             # 42
    dists   = pairwise_dists(pts_canon)       # 55
    curls   = finger_curls(pts_canon)         # 5
    feats = np.concatenate([xy_flat, dists, curls], axis=0)  # 102 dims
    return feats.astype(np.float32)
 