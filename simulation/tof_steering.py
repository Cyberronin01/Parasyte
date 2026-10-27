"""
PARASYTE — Reactive Optical Time-of-Flight (ToF) Obstacle Avoidance Teacher.

Operates purely on the front VL53L5CX 64-zone (8x8) depth matrix:
- Uses NO global world coordinates.
- Uses NO omniscient obstacle lookups.
- Divides the 45-degree forward optical cone into Left, Center, and Right sectors.
- Computes safe steering commands [pwm_left, pwm_right] to steer away from obstacles.
"""

import numpy as np
from typing import Tuple


def compute_reactive_tof_steering(tof_depth_64: np.ndarray,
                                  base_speed: float = 0.65,
                                  steer_intensity: float = 0.40) -> Tuple[float, float, str]:
    """
    Computes safe reactive steering PWMs from raw 64-beam ToF matrix.
    tof_depth_64: (64,) array in [0.0, 1.0] (1.0 = 4000 mm).
    Returns:
        (pwm_left, pwm_right, decision_str)
    """
    # Reshape into 8 rows x 8 columns
    grid = tof_depth_64.reshape(8, 8)

    # Columns:
    # 0, 1, 2: Left sector (-22.5 deg to -7.5 deg)
    # 3, 4:    Center sector (-7.5 deg to +7.5 deg)
    # 5, 6, 7: Right sector (+7.5 deg to +22.5 deg)

    # Focus on the middle rows (rows 2 to 6, avoiding ground reflections in row 7)
    mid_grid = grid[2:6, :]

    d_left = float(np.min(mid_grid[:, 0:3]))
    d_center = float(np.min(mid_grid[:, 3:5]))
    d_right = float(np.min(mid_grid[:, 5:8]))

    # Thresholds (normalized to 4000 mm):
    # 0.25 = 1000 mm, 0.15 = 600 mm, 0.08 = 320 mm
    CRITICAL_DIST = 0.10   # 400 mm
    WARNING_DIST = 0.25    # 1000 mm

    # Case 1: Critical Proximity directly ahead (Sharp Evasive Arc)
    if d_center < CRITICAL_DIST:
        if d_left >= d_right:
            # Curve sharply Left: slow left track to 0.15, right track drives at 0.75
            return 0.15, 0.75, "EVADE_LEFT"
        else:
            # Curve sharply Right: left track drives at 0.75, slow right track to 0.15
            return 0.75, 0.15, "EVADE_RIGHT"

    # Case 2: Obstacle detected ahead in warning zone (Proportional steering)
    if d_center < WARNING_DIST:
        if d_left > d_right:
            # Gentle Left arc: slow down left slightly, speed up right
            pwm_l = max(0.30, base_speed - steer_intensity * 0.7)
            pwm_r = min(0.85, base_speed + steer_intensity * 0.5)
            return pwm_l, pwm_r, "STEER_LEFT"
        else:
            # Gentle Right arc: speed up left, slow down right
            pwm_l = min(0.85, base_speed + steer_intensity * 0.5)
            pwm_r = max(0.30, base_speed - steer_intensity * 0.7)
            return pwm_l, pwm_r, "STEER_RIGHT"

    # Case 3: Glancing obstacle on Left sector
    if d_left < WARNING_DIST * 0.7:
        # Gently steer right
        return base_speed + 0.15, base_speed - 0.20, "AVOID_LEFT_WALL"

    # Case 4: Glancing obstacle on Right sector
    if d_right < WARNING_DIST * 0.7:
        # Gently steer left
        return base_speed - 0.20, base_speed + 0.15, "AVOID_RIGHT_WALL"

    # Case 5: Path Clear -> Straight forward walking
    return base_speed, base_speed, "CLEAR_AHEAD"
