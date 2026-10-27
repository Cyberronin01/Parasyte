"""
PARASYTE Autonomous Governance Engine & Sim-to-Real Supervisor
=============================================================
Orchestrates intrinsic skill discovery, trial evaluation, EWC consolidation,
hardware safety guardrails, and runtime active goal arbitration.
Works symmetrically in SITL simulation and on physical hardware.
"""

import math
import numpy as np
import torch
from typing import Dict, Any, Tuple, Optional, List

from .hardware_guardrails import HardwareSafetyGuard
from .skill_contract import (
    ForwardSkillContract, BackwardSkillContract, SteeringSkillContract,
    TrialObservation, GOAL_FORWARD, GOAL_BACKWARD, GOAL_STEER_LEFT, GOAL_STEER_RIGHT,
    get_goal_vector, GOAL_NAMES
)
from .mastery_evaluator import MasteryEvaluator, SkillStatus


class GovernanceEngine:
    """
    Master Governor managing skill contracts, physical certification trials,
    real-time hardware interlocks, and active inference goal selection.
    """
    def __init__(self, brain, hal, guardrails: Optional[HardwareSafetyGuard] = None, dt: float = 0.02):
        self.brain = brain
        self.hal = hal
        self.guard = guardrails or HardwareSafetyGuard(max_slew=3.0, dt=dt)
        self.evaluator = MasteryEvaluator()
        self.dt = dt

        # Register Core Skill Tree with 100% Mastery Prerequisites
        self.skill_fwd = self.evaluator.register_skill(ForwardSkillContract())
        self.skill_rev = self.evaluator.register_skill(BackwardSkillContract())
        self.skill_steer_l = self.evaluator.register_skill(
            SteeringSkillContract(is_left=True),
            prerequisites=["locomotion.forward", "locomotion.reverse"]
        )
        self.skill_steer_r = self.evaluator.register_skill(
            SteeringSkillContract(is_left=False),
            prerequisites=["locomotion.forward", "locomotion.reverse"]
        )

        # Runtime State
        self.active_goal = GOAL_FORWARD
        self.step_count = 0
        self.stats = {
            "reverse_disengagements": 0,
            "safety_interlocks": 0,
            "certified_skills": []
        }

    def run_trial(self, goal_id: int, steps: int = 50) -> TrialObservation:
        """Runs an isolated evaluation trial using only onboard sensor telemetry."""
        trial = TrialObservation()
        g_vec = torch.from_numpy(get_goal_vector(goal_id))

        for _ in range(steps):
            # 1. Onboard sensor read
            pkt = self.hal.get_latest_packet() if hasattr(self.hal, "get_latest_packet") else None
            if pkt is None:
                # Step with zero to obtain packet if first step
                pkt = self.hal.step(0.0, 0.0)

            t_depth = torch.from_numpy(pkt["depth"])
            t_depth_rear = torch.from_numpy(pkt["depth_rear"]) if "depth_rear" in pkt else None
            t_imu = torch.from_numpy(pkt["imu"])
            t_enc = torch.from_numpy(pkt["encoders"])
            t_mot = torch.from_numpy(pkt["motor_echo"])

            with torch.no_grad():
                emb, _ = self.brain.perception(t_depth, t_imu, t_enc, t_mot, t_depth_rear)
                st, _ = self.brain.fusion(emb)
                act = self.brain.act(st, g_vec)

            raw_l = float(np.clip(act[0].item(), -1.0, 1.0))
            raw_r = float(np.clip(act[1].item(), -1.0, 1.0))

            # Hardware guardrail filtering
            accel = (float(pkt["imu"][0]), float(pkt["imu"][1]), float(pkt["imu"][2]))
            speeds = (float(pkt["encoders"][0]), float(pkt["encoders"][1]))
            safe_l, safe_r, info = self.guard.filter_action(raw_l, raw_r, accel, speeds)

            pkt = self.hal.step(safe_l, safe_r)

            # Record telemetry strictly from onboard sensors
            enc_disp = float((pkt["encoders"][0] + pkt["encoders"][1]) * 0.5 * 1000.0 * self.dt)
            yaw_rate = float(pkt["imu"][5])
            min_clr = float(np.min(pkt["depth"]))
            is_stall = info.get("interlock", False)
            is_coll = bool(pkt.get("bumper_contact", False))

            trial.record_step(enc_disp, yaw_rate, min_clr, is_stall, is_coll)

        return trial

    def arbitrate_goal(self, unified_state: torch.Tensor, current_depth: torch.Tensor) -> int:
        """
        Active Inference Goal Selector (Zero Heuristics):
        Mentally simulates available unlocked skills through the Action-Conditioned World Model
        and selects the goal token that minimizes Expected Free Energy G(g).
        """
        # Filter available goals based on certified skill hierarchy
        unlocked_goals = []
        for name, prog in self.evaluator.skills.items():
            if prog.status != SkillStatus.LOCKED:
                unlocked_goals.append(prog.contract.goal_id)

        if not unlocked_goals:
            unlocked_goals = [GOAL_FORWARD]

        best_goal, _ = self.brain.select_skill(
            unified_state=unified_state,
            current_depth=current_depth,
            available_goals=unlocked_goals
        )
        if best_goal == GOAL_BACKWARD and self.active_goal != GOAL_BACKWARD:
            self.stats["reverse_disengagements"] += 1
        self.active_goal = best_goal
        return best_goal
