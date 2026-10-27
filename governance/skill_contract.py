"""
PARASYTE Skill Contracts & Sensory Grounded Objectives
======================================================
Defines the sensory contracts for all developmental skill primitives.
Zero motor PWM values or human-assigned cheats are permitted.
All objectives and evaluations are grounded strictly in onboard sensor telemetry:
- IMU linear acceleration & angular gyro yaw rate
- Magnetic quadrature wheel/crank encoders
- 64-beam optical Time-of-Flight matrix
"""

import math
from typing import Dict, Any, List, Tuple
import numpy as np


# 4-Dimensional Standardized Goal Tokens
GOAL_FORWARD = 0
GOAL_BACKWARD = 1
GOAL_STEER_LEFT = 2
GOAL_STEER_RIGHT = 3
GOAL_DIM = 4

GOAL_NAMES = {
    GOAL_FORWARD: "locomotion.forward",
    GOAL_BACKWARD: "locomotion.reverse",
    GOAL_STEER_LEFT: "steering.left",
    GOAL_STEER_RIGHT: "steering.right",
}


def get_goal_vector(goal_id: int) -> np.ndarray:
    """Returns a 4-dimensional one-hot goal embedding."""
    vec = np.zeros(GOAL_DIM, dtype=np.float32)
    vec[goal_id] = 1.0
    return vec


class TrialObservation:
    """Logs onboard telemetry during an evaluation trial."""
    def __init__(self):
        self.encoder_displacements: List[float] = []
        self.yaw_rates: List[float] = []
        self.min_clearances: List[float] = []
        self.stall_events: int = 0
        self.collision_events: int = 0

    def record_step(self, enc_disp: float, yaw_rate: float, min_clearance: float, is_stall: bool, is_collision: bool):
        self.encoder_displacements.append(enc_disp)
        self.yaw_rates.append(yaw_rate)
        self.min_clearances.append(min_clearance)
        if is_stall:
            self.stall_events += 1
        if is_collision:
            self.collision_events += 1


class SkillContract:
    """Abstract base contract defining what success means for an embodied skill."""
    def __init__(self, name: str, goal_id: int, description: str):
        self.name = name
        self.goal_id = goal_id
        self.description = description

    def evaluate_trial(self, trial: TrialObservation) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Evaluates an empirical trial.
        Returns:
            (passed: bool, score: float in [0.0, 1.0], diagnostics: dict)
        """
        raise NotImplementedError


class ForwardSkillContract(SkillContract):
    """
    Contract for Forward Locomotion.
    Success Criteria:
    - Consistent positive encoder velocity (avg >= 35.0 mm/s)
    - Cumulative integrated yaw drift <= 2.0 degrees
    - Zero sustained motor stall events
    """
class ForwardSkillContract(SkillContract):
    """
    Contract for Forward Locomotion.
    Supports multi-speed certification (crawl, cruise, sprint).
    Success Criteria (100% Mastery Gauntlet):
    - Consistent positive encoder velocity (avg >= min_speed mm/s)
    - Cumulative integrated yaw drift <= max_drift_deg
    - Zero sustained motor stall events
    """
    def __init__(self, min_speed: float = 30.0, max_drift_deg: float = 1.0):
        super().__init__(
            name="locomotion.forward",
            goal_id=GOAL_FORWARD,
            description="Autonomous straight forward rhythmic gait"
        )
        self.min_speed = min_speed
        self.max_drift_deg = max_drift_deg

    def evaluate_trial(self, trial: TrialObservation) -> Tuple[bool, float, Dict[str, Any]]:
        if not trial.encoder_displacements:
            return False, 0.0, {"error": "empty trial"}

        total_disp = sum(trial.encoder_displacements)  # mm
        duration_s = len(trial.encoder_displacements) * 0.02
        avg_speed = total_disp / max(0.01, duration_s)

        # Integrated gyro yaw drift
        total_drift_deg = math.degrees(sum(abs(yr) * 0.02 for yr in trial.yaw_rates))

        passed = (
            avg_speed >= self.min_speed and
            total_drift_deg <= self.max_drift_deg and
            trial.stall_events == 0
        )

        speed_score = min(1.0, max(0.0, avg_speed / max(self.min_speed, 1.0)))
        drift_score = min(1.0, max(0.0, 1.0 - (max(0.0, total_drift_deg - (self.max_drift_deg * 0.8)) / 2.0)))
        stall_penalty = 0.5 if trial.stall_events > 0 else 1.0

        score = float(speed_score * 0.6 + drift_score * 0.4) * stall_penalty

        return passed, score, {
            "avg_speed_mm_s": round(avg_speed, 1),
            "total_drift_deg": round(total_drift_deg, 2),
            "stall_events": trial.stall_events,
            "passed": passed
        }


class BackwardSkillContract(SkillContract):
    """
    Contract for Reverse Locomotion.
    Supports multi-speed certification (crawl, cruise, sprint).
    Success Criteria (100% Mastery Gauntlet):
    - Consistent negative encoder velocity (avg <= min_reverse_speed mm/s)
    - Cumulative integrated yaw drift <= max_drift_deg
    - Zero sustained motor stall events
    """
    def __init__(self, min_reverse_speed: float = -25.0, max_drift_deg: float = 1.2):
        super().__init__(
            name="locomotion.reverse",
            goal_id=GOAL_BACKWARD,
            description="Autonomous straight backward rhythmic gait"
        )
        self.min_reverse_speed = min_reverse_speed
        self.max_drift_deg = max_drift_deg

    def evaluate_trial(self, trial: TrialObservation) -> Tuple[bool, float, Dict[str, Any]]:
        if not trial.encoder_displacements:
            return False, 0.0, {"error": "empty trial"}

        total_disp = sum(trial.encoder_displacements)  # negative mm
        duration_s = len(trial.encoder_displacements) * 0.02
        avg_speed = total_disp / max(0.01, duration_s)

        total_drift_deg = math.degrees(sum(abs(yr) * 0.02 for yr in trial.yaw_rates))

        passed = (
            avg_speed <= self.min_reverse_speed and
            total_drift_deg <= self.max_drift_deg and
            trial.stall_events == 0
        )

        speed_score = min(1.0, max(0.0, abs(avg_speed) / max(abs(self.min_reverse_speed), 1.0)))
        drift_score = min(1.0, max(0.0, 1.0 - (max(0.0, total_drift_deg - (self.max_drift_deg * 0.8)) / 2.0)))
        stall_penalty = 0.5 if trial.stall_events > 0 else 1.0

        score = float(speed_score * 0.6 + drift_score * 0.4) * stall_penalty

        return passed, score, {
            "avg_reverse_speed_mm_s": round(avg_speed, 1),
            "total_drift_deg": round(total_drift_deg, 2),
            "stall_events": trial.stall_events,
            "passed": passed
        }


class SteeringSkillContract(SkillContract):
    """
    Contract for Directional Steering.
    Success Criteria:
    - Achieves commanded angular velocity (gyro yaw rate) in requested direction
    - Minimal lateral skid stall
    """
    def __init__(self, is_left: bool = True):
        name = "steering.left" if is_left else "steering.right"
        goal_id = GOAL_STEER_LEFT if is_left else GOAL_STEER_RIGHT
        super().__init__(name=name, goal_id=goal_id, description=f"Directional {name} control")
        self.is_left = is_left

    def evaluate_trial(self, trial: TrialObservation) -> Tuple[bool, float, Dict[str, Any]]:
        if not trial.yaw_rates:
            return False, 0.0, {"error": "empty trial"}

        # Commanded direction: Left = positive yaw rate, Right = negative yaw rate
        expected_sign = 1.0 if self.is_left else -1.0
        signed_rates = [yr * expected_sign for yr in trial.yaw_rates]
        mean_rate = sum(signed_rates) / len(signed_rates)

        passed = mean_rate >= 0.25 and trial.stall_events <= 1

        # Target turning rate for 8-legged chassis: >= 0.5 rad/s (~28.6 deg/s)
        rate_score = min(1.0, max(0.0, mean_rate / 0.5))
        score = float(rate_score)

        return passed, score, {
            "mean_yaw_rate_rad_s": round(mean_rate, 2),
            "stall_events": trial.stall_events,
            "passed": passed
        }
