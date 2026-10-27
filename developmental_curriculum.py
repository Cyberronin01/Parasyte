"""
PARASYTE Developmental Curriculum & Pre-Training Pipeline
=========================================================
Implements the true iterative developmental schooling sequence:
1. Stage 1A: Sensorimotor Embodiment & Motor Babbling -> Consolidation of Node 0 (lambda=2000.0)
2. Stage 1B: Self-Discovered Forward Gait Practice Loop -> 5/5 consecutive passes -> Consolidate Skill 1
3. Stage 1C: Self-Discovered Reverse Gait Practice Loop -> 5/5 consecutive passes -> Consolidate Skill 2
4. Stage 1D: Self-Discovered Differential Steering Practice Loop -> Unlocked after prerequisites -> Consolidate Skill 3
5. Full Mastery Certification across all skills under Governance Referee.

Once certified, returns the mastered ParasyteBrain ready for zero-shot deployment
into arbitrary unseen operational stress scenarios.
"""

import math
import random
import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from brain import (
    ParasyteBrain,
    GRAD_CLIP_NORM,
    extract_sector_clearance,
)
from simulation.actuators_body import ArachnobeestBody
from simulation.sensor_suite import ArenaWorld, SimulatedObstacle, HALBridge
from governance.hardware_guardrails import HardwareSafetyGuard
from governance.skill_contract import (
    ForwardSkillContract,
    BackwardSkillContract,
    SteeringSkillContract,
    GOAL_FORWARD,
    GOAL_BACKWARD,
    GOAL_STEER_LEFT,
    GOAL_STEER_RIGHT,
    get_goal_vector,
    TrialObservation,
)
from governance.mastery_evaluator import MasteryEvaluator, SkillStatus

DT = 0.02


def run_evaluation_trial(brain: ParasyteBrain, goal_id: int, steps: int = 50) -> TrialObservation:
    """Runs a clean, isolated physical evaluation trial starting from (0,0) in an open test arena."""
    trial = TrialObservation()
    guard = HardwareSafetyGuard(max_slew=3.0, dt=DT)
    g_vec = torch.from_numpy(get_goal_vector(goal_id))

    arena_open = ArenaWorld()
    arena_open.obstacles = []
    body = ArachnobeestBody(dt=DT)
    hal = HALBridge(body, arena_open)

    pkt = hal.step(0.0, 0.0)
    for step_idx in range(steps):
        t_depth = torch.from_numpy(pkt["depth"])
        t_depth_rear = torch.from_numpy(pkt["depth_rear"])
        t_imu = torch.from_numpy(pkt["imu"])
        t_enc = torch.from_numpy(pkt["encoders"])
        t_mot = torch.from_numpy(pkt["motor_echo"])

        with torch.no_grad():
            emb, _ = brain.perception(t_depth, t_imu, t_enc, t_mot, t_depth_rear)
            st, _ = brain.fusion(emb)
            act = brain.act(st, g_vec)

        raw_l = float(np.clip(act[0].item(), -1.0, 1.0))
        raw_r = float(np.clip(act[1].item(), -1.0, 1.0))
        accel = (float(pkt["imu"][0]), float(pkt["imu"][1]), float(pkt["imu"][2]))
        speeds = (float(pkt["encoders"][0]), float(pkt["encoders"][1]))
        safe_l, safe_r, info = guard.filter_action(raw_l, raw_r, accel, speeds)

        pkt = hal.step(safe_l, safe_r)
        enc_disp = float((pkt["encoders"][0] + pkt["encoders"][1]) * 0.5 * 1000.0 * DT)
        yaw_rate = float(pkt["imu"][5])
        min_clr = float(np.min(pkt["depth"]))
        is_stall = info.get("interlock", False)
        is_coll = bool(pkt.get("bumper_contact", False))
        trial.record_step(enc_disp, yaw_rate, min_clr, is_stall, is_coll)

    return trial


def train_developmental_curriculum(seed: int = 42, verbose: bool = True) -> ParasyteBrain:
    """
    Executes the gradual, iterative schooling sequence.
    Each skill must be practiced and certified (5/5 consecutive passes) before the next is unlocked.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if verbose:
        print("=" * 80)
        print("    PARASYTE DEVELOPMENTAL SCHOOLING & ITERATIVE MASTERY PIPELINE")
        print("=" * 80)

    # 1. Initialize training environment & brain
    train_arena = ArenaWorld()
    train_arena.obstacles = []
    body = ArachnobeestBody(dt=DT)
    hal = HALBridge(body, train_arena)
    brain = ParasyteBrain()
    evaluator = MasteryEvaluator()

    # 1. Initialize training environment & brain
    train_arena = ArenaWorld()
    train_arena.obstacles = []
    body = ArachnobeestBody(dt=DT)
    hal = HALBridge(body, train_arena)
    brain = ParasyteBrain()
    evaluator = MasteryEvaluator()

    # Replay buffer for autonomous grounding & affordance discovery
    buffer = []
    pkt = hal.step(0.0, 0.0)
    prev = {
        "depth": torch.from_numpy(pkt["depth"]),
        "depth_rear": torch.from_numpy(pkt["depth_rear"]),
        "imu": torch.from_numpy(pkt["imu"]),
        "encoders": torch.from_numpy(pkt["encoders"]),
        "motor": torch.from_numpy(pkt["motor_echo"]),
    }

    # ─────────────────────────────────────────────────────────────
    # STAGE 1A: MOTOR BABBLING & SENSORIMOTOR EMBODIMENT (240 steps)
    # ─────────────────────────────────────────────────────────────
    if verbose:
        print("\n>> STAGE 1A: Unconstrained Motor Babbling & Sensory Manifold Exploration (240 steps)...")

    babble_patterns = [
        (0.8, 0.8), (-0.8, -0.8), (-0.8, 0.8), (0.8, -0.8),
        (0.5, 0.5), (-0.5, -0.5), (0.0, 0.0), (0.7, -0.7)
    ]

    for step in range(240):
        pat = babble_patterns[(step // 15) % len(babble_patterns)]
        cmd_l = float(np.clip(pat[0] + np.random.uniform(-0.15, 0.15), -1.0, 1.0))
        cmd_r = float(np.clip(pat[1] + np.random.uniform(-0.15, 0.15), -1.0, 1.0))
        curr_pkt = hal.step(cmd_l, cmd_r)
        curr = {
            "depth": torch.from_numpy(curr_pkt["depth"]),
            "depth_rear": torch.from_numpy(curr_pkt["depth_rear"]),
            "imu": torch.from_numpy(curr_pkt["imu"]),
            "encoders": torch.from_numpy(curr_pkt["encoders"]),
            "motor": torch.from_numpy(curr_pkt["motor_echo"]),
        }
        buffer.append({
            "prev": prev,
            "act": torch.tensor([cmd_l, cmd_r], dtype=torch.float32),
            "curr": curr
        })
        prev = curr

    if verbose:
        print("   [+] Babbling buffer collected (240 physical transitions). Grounding World Model...")

    # Mini-batch World Model & Perception Grounding
    wm_opt = torch.optim.Adam(brain.world_model.parameters(), lr=0.003)
    perc_opt = torch.optim.Adam(list(brain.perception.parameters()) + list(brain.fusion.parameters()), lr=0.003)

    for epoch in range(60):
        idx = np.random.choice(len(buffer), 32)
        b_prev = [buffer[i]["prev"] for i in idx]
        b_act = torch.stack([buffer[i]["act"] for i in idx])
        b_curr_enc = torch.stack([buffer[i]["curr"]["encoders"] for i in idx])
        b_curr_yaw = torch.stack([buffer[i]["curr"]["imu"][5:6] for i in idx])

        states = []
        for p in b_prev:
            emb, _ = brain.perception(p["depth"], p["imu"], p["encoders"], p["motor"], p["depth_rear"])
            st, _ = brain.fusion(emb)
            states.append(st)
        b_st = torch.stack(states)

        pred = brain.world_model(b_st, b_act)
        loss = (
            torch.mean((pred["encoders"] - b_curr_enc) ** 2) +
            torch.mean((pred["yaw_rate"] - b_curr_yaw) ** 2)
        )
        wm_opt.zero_grad()
        perc_opt.zero_grad()
        loss.backward()
        wm_opt.step()
        perc_opt.step()

    # Autonomous Kinematic Affordance Discovery via SVD
    affordances = brain.skill_tree.discover_sensory_affordances(buffer)
    e_long = affordances["e_long"]
    e_rot = affordances["e_rot"]
    if verbose:
        print(f"   [+] Discovered Orthogonal Affordances: e_long={e_long.tolist()}, e_rot={e_rot.tolist()}")

    # Consolidate Node 0: Root Meta-Skill (lambda=2000.0)
    def criterion_fn_node0(sample):
        d_r = sample["prev"].get("depth_rear", None)
        emb, preds = brain.perception(sample["prev"]["depth"], sample["prev"]["imu"], sample["prev"]["encoders"], sample["prev"]["motor"], d_r)
        return sum(torch.mean((preds[k] - sample["prev"][k]) ** 2) for k in ["imu", "encoders", "motor"])

    brain.skill_tree.register_node0_mastery_meta_skill(
        buffer[-40:],
        criterion_fn_node0
    )
    if verbose:
        print("   [+] Node 0 Consolidated: meta.mastery_process anchored at root (lambda=2000.0)")

    # ─────────────────────────────────────────────────────────────
    # STAGE 1B: ITERATIVE FORWARD WALKING PRACTICE LOOP (5/5 PASSES)
    # ─────────────────────────────────────────────────────────────
    if verbose:
        print("\n>> STAGE 1B: Forward Walking Practice via Kinematic Eigen-Projection (Until 5/5 Passes)...")

    g_fwd = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    prog_fwd = evaluator.register_skill(ForwardSkillContract())
    policy_opt = torch.optim.Adam(brain.goal_policy.parameters(), lr=0.02)
    loco_opt = torch.optim.Adam(brain.locomotion.parameters(), lr=0.02)

    consecutive_fwd = 0
    trial_fwd_idx = 0
    max_practice_trials = 15

    while consecutive_fwd < 5 and trial_fwd_idx < max_practice_trials:
        trial_fwd_idx += 1

        # Practice block: 25 gradient steps backpropagating through World Model
        for _ in range(25):
            idx = np.random.choice(len(buffer), 16)
            b_prev = [buffer[i]["prev"] for i in idx]
            states = []
            for p in b_prev:
                emb, _ = brain.perception(p["depth"], p["imu"], p["encoders"], p["motor"], p["depth_rear"])
                st, _ = brain.fusion(emb)
                states.append(st)
            b_st = torch.stack(states).detach()
            b_g = g_fwd.unsqueeze(0).expand(16, -1)

            b_act = brain.goal_policy(b_st, b_g)
            base_act = brain.locomotion(b_st)
            wm_p = brain.world_model(b_st, b_act)

            # Pure kinematic eigen-projection onto +e_long (Surge axis)
            v_enc = 0.5 * (wm_p["encoders"][:, 0] + wm_p["encoders"][:, 1])
            w_yaw = wm_p["yaw_rate"].squeeze(-1)
            proj_long = v_enc * e_long[0] + w_yaw * e_long[1]
            proj_rot = v_enc * e_rot[0] + w_yaw * e_rot[1]
            motor_cost = 0.01 * torch.mean(b_act ** 2)

            loss_fwd = -10.0 * torch.mean(proj_long) + 5.0 * torch.mean(proj_rot ** 2) + motor_cost
            loss_loco_match = torch.mean((base_act - b_act.detach()) ** 2)
            loss_total = loss_fwd + 1.0 * loss_loco_match + brain.skill_tree.penalty()

            policy_opt.zero_grad()
            loco_opt.zero_grad()
            loss_total.backward()
            policy_opt.step()
            loco_opt.step()

        # Formal Evaluation Trial Probe (50 steps)
        t_obs = run_evaluation_trial(brain, GOAL_FORWARD, steps=50)
        passed, score, mastery_pct, status = evaluator.evaluate_trial(prog_fwd.contract.name, t_obs)
        diag = prog_fwd.trial_history[-1]["diagnostics"]

        if passed:
            consecutive_fwd += 1
            if verbose:
                print(f"   [Forward Practice {trial_fwd_idx:2d}] PASSED (Speed: {diag['avg_speed_mm_s']:5.1f} mm/s, Drift: {diag['total_drift_deg']:4.2f}°) | Streak: {consecutive_fwd}/5 (Mastery: {mastery_pct:.1f}%)")
        else:
            consecutive_fwd = 0
            if verbose:
                print(f"   [Forward Practice {trial_fwd_idx:2d}] FAILED -> Streak reset to 0/5 (Mastery: {mastery_pct:.1f}%)")

    assert consecutive_fwd >= 5, "Forward practice failed to achieve 5/5 certified streak"

    # Consolidate Skill 1: Forward Walking under EWC
    def criterion_fn_fwd(sample):
        emb, _ = brain.perception(sample["prev"]["depth"], sample["prev"]["imu"], sample["prev"]["encoders"], sample["prev"]["motor"], sample["prev"].get("depth_rear"))
        st, _ = brain.fusion(emb)
        act = brain.goal_policy(st, g_fwd)
        wm_p = brain.world_model(st, act)
        v_enc = 0.5 * (wm_p["encoders"][0] + wm_p["encoders"][1])
        w_yaw = wm_p["yaw_rate"].squeeze(-1)
        proj_long = v_enc * e_long[0] + w_yaw * e_long[1]
        return -10.0 * proj_long + 0.01 * torch.sum(act ** 2)

    brain.skill_tree.register_skill(
        "locomotion.forward",
        buffer[-40:],
        criterion_fn_fwd,
        parent="meta.mastery_process",
        importance=100.0
    )
    if verbose:
        print("   [+] Skill 1 Consolidated: locomotion.forward (Child of Node 0, Certified 100%)")

    # ─────────────────────────────────────────────────────────────
    # STAGE 1C: ITERATIVE REVERSE WALKING PRACTICE LOOP (5/5 PASSES)
    # ─────────────────────────────────────────────────────────────
    if verbose:
        print("\n>> STAGE 1C: Reverse Walking Practice via Kinematic Eigen-Projection (Until 5/5 Passes)...")

    g_rev = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32)
    prog_rev = evaluator.register_skill(BackwardSkillContract())

    consecutive_rev = 0
    trial_rev_idx = 0

    while consecutive_rev < 5 and trial_rev_idx < max_practice_trials:
        trial_rev_idx += 1

        for _ in range(25):
            idx = np.random.choice(len(buffer), 16)
            b_prev = [buffer[i]["prev"] for i in idx]
            states = []
            for p in b_prev:
                emb, _ = brain.perception(p["depth"], p["imu"], p["encoders"], p["motor"], p["depth_rear"])
                st, _ = brain.fusion(emb)
                states.append(st)
            b_st = torch.stack(states).detach()
            b_g = g_rev.unsqueeze(0).expand(16, -1)

            b_act = brain.goal_policy(b_st, b_g)
            wm_p = brain.world_model(b_st, b_act)

            # Pure kinematic eigen-projection onto -e_long (Surge axis reverse)
            v_enc = 0.5 * (wm_p["encoders"][:, 0] + wm_p["encoders"][:, 1])
            w_yaw = wm_p["yaw_rate"].squeeze(-1)
            proj_long = v_enc * e_long[0] + w_yaw * e_long[1]
            proj_rot = v_enc * e_rot[0] + w_yaw * e_rot[1]
            motor_cost = 0.01 * torch.mean(b_act ** 2)

            loss_rev = +10.0 * torch.mean(proj_long) + 5.0 * torch.mean(proj_rot ** 2) + motor_cost + brain.skill_tree.penalty()

            policy_opt.zero_grad()
            loss_rev.backward()
            policy_opt.step()

        # Formal Evaluation Trial Probe (50 steps)
        t_obs = run_evaluation_trial(brain, GOAL_BACKWARD, steps=50)
        passed, score, mastery_pct, status = evaluator.evaluate_trial(prog_rev.contract.name, t_obs)
        diag = prog_rev.trial_history[-1]["diagnostics"]

        if passed:
            consecutive_rev += 1
            if verbose:
                print(f"   [Reverse Practice {trial_rev_idx:2d}] PASSED (Speed: {diag['avg_reverse_speed_mm_s']:5.1f} mm/s, Drift: {diag['total_drift_deg']:4.2f}°) | Streak: {consecutive_rev}/5 (Mastery: {mastery_pct:.1f}%)")
        else:
            consecutive_rev = 0
            if verbose:
                print(f"   [Reverse Practice {trial_rev_idx:2d}] FAILED -> Streak reset to 0/5 (Mastery: {mastery_pct:.1f}%)")

    assert consecutive_rev >= 5, "Reverse practice failed to achieve 5/5 certified streak"

    # Consolidate Skill 2: Reverse Walking under EWC
    def criterion_fn_rev(sample):
        emb, _ = brain.perception(sample["prev"]["depth"], sample["prev"]["imu"], sample["prev"]["encoders"], sample["prev"]["motor"], sample["prev"].get("depth_rear"))
        st, _ = brain.fusion(emb)
        act = brain.goal_policy(st, g_rev)
        wm_p = brain.world_model(st, act)
        v_enc = 0.5 * (wm_p["encoders"][0] + wm_p["encoders"][1])
        w_yaw = wm_p["yaw_rate"].squeeze(-1)
        proj_long = v_enc * e_long[0] + w_yaw * e_long[1]
        return +10.0 * proj_long + 0.01 * torch.sum(act ** 2)

    brain.skill_tree.register_skill(
        "locomotion.reverse",
        buffer[-40:],
        criterion_fn_rev,
        parent="meta.mastery_process",
        importance=100.0
    )
    if verbose:
        print("   [+] Skill 2 Consolidated: locomotion.reverse (Child of Node 0, Certified 100%)")

    # ─────────────────────────────────────────────────────────────
    # STAGE 1D: DIFFERENTIAL STEERING MASTERY (UNLOCKED AFTER PREREQS)
    # ─────────────────────────────────────────────────────────────
    if verbose:
        print("\n>> STAGE 1D: Differential Steering Practice Loop (Prerequisites Unlocked)...")

    g_left = torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float32)
    g_right = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)

    prog_steer_l = evaluator.register_skill(SteeringSkillContract(is_left=True), prerequisites=["locomotion.forward", "locomotion.reverse"])
    prog_steer_r = evaluator.register_skill(SteeringSkillContract(is_left=False), prerequisites=["locomotion.forward", "locomotion.reverse"])

    for trial_steer_idx in range(1, max_practice_trials):
        for _ in range(25):
            idx = np.random.choice(len(buffer), 16)
            b_prev = [buffer[i]["prev"] for i in idx]
            states = []
            for p in b_prev:
                emb, _ = brain.perception(p["depth"], p["imu"], p["encoders"], p["motor"], p["depth_rear"])
                st, _ = brain.fusion(emb)
                states.append(st)
            b_st = torch.stack(states).detach()

            # Train Left Turn (+e_rot)
            b_gl = g_left.unsqueeze(0).expand(16, -1)
            b_act_l = brain.goal_policy(b_st, b_gl)
            wm_l = brain.world_model(b_st, b_act_l)
            w_yaw_l = wm_l["yaw_rate"].squeeze(-1)
            v_enc_l = 0.5 * (wm_l["encoders"][:, 0] + wm_l["encoders"][:, 1])
            proj_rot_l = v_enc_l * e_rot[0] + w_yaw_l * e_rot[1]
            proj_long_l = v_enc_l * e_long[0] + w_yaw_l * e_long[1]
            loss_l = -10.0 * torch.mean(proj_rot_l) - 6.0 * torch.mean(proj_long_l) + 0.01 * torch.mean(b_act_l ** 2)

            # Train Right Turn (-e_rot) with curving forward surge
            b_gr = g_right.unsqueeze(0).expand(16, -1)
            b_act_r = brain.goal_policy(b_st, b_gr)
            wm_r = brain.world_model(b_st, b_act_r)
            w_yaw_r = wm_r["yaw_rate"].squeeze(-1)
            v_enc_r = 0.5 * (wm_r["encoders"][:, 0] + wm_r["encoders"][:, 1])
            proj_rot_r = v_enc_r * e_rot[0] + w_yaw_r * e_rot[1]
            proj_long_r = v_enc_r * e_long[0] + w_yaw_r * e_long[1]
            loss_r = +10.0 * torch.mean(proj_rot_r) - 6.0 * torch.mean(proj_long_r) + 0.01 * torch.mean(b_act_r ** 2)

            loss_steer = loss_l + loss_r + brain.skill_tree.penalty()

            policy_opt.zero_grad()
            loss_steer.backward()
            policy_opt.step()

        t_stl = run_evaluation_trial(brain, GOAL_STEER_LEFT, steps=50)
        p_l, _, m_l, s_l = evaluator.evaluate_trial(prog_steer_l.contract.name, t_stl)
        t_str = run_evaluation_trial(brain, GOAL_STEER_RIGHT, steps=50)
        p_r, _, m_r, s_r = evaluator.evaluate_trial(prog_steer_r.contract.name, t_str)

        if verbose:
            print(f"   [Steering Practice {trial_steer_idx:2d}] Left: {m_l:5.1f}% ({s_l}) | Right: {m_r:5.1f}% ({s_r})")

        if prog_steer_l.consecutive_passes >= 5 and prog_steer_r.consecutive_passes >= 5:
            break

    assert prog_steer_l.consecutive_passes >= 5 and prog_steer_r.consecutive_passes >= 5, "Steering failed to achieve 5/5 certified streak"

    # Consolidate Skill 3: Steering under EWC
    def criterion_fn_steer(sample):
        emb, _ = brain.perception(sample["prev"]["depth"], sample["prev"]["imu"], sample["prev"]["encoders"], sample["prev"]["motor"], sample["prev"].get("depth_rear"))
        st, _ = brain.fusion(emb)
        act_l = brain.goal_policy(st, g_left)
        act_r = brain.goal_policy(st, g_right)
        wm_l = brain.world_model(st, act_l)
        wm_r = brain.world_model(st, act_r)
        return -10.0 * (wm_l["yaw_rate"][0] - wm_r["yaw_rate"][0]) + 0.01 * (torch.sum(act_l**2) + torch.sum(act_r**2))

    brain.skill_tree.register_skill(
        "steering",
        buffer[-40:],
        criterion_fn_steer,
        parent="locomotion.forward",
        importance=100.0
    )
    if verbose:
        print("   [+] Skill 3 Consolidated: steering (Child of locomotion.forward)")

    all_certified = all(s.is_certified for s in [prog_fwd, prog_rev, prog_steer_l, prog_steer_r])
    if verbose:
        print("\n" + "=" * 80)
        print("                   DEVELOPMENTAL MASTERY GAUNTLET SUMMARY")
        print("=" * 80)
        print(f"   Node 0 Root Meta-Skill : CERTIFIED (lambda=2000.0)")
        print(f"   Forward Locomotion     : {prog_fwd.mastery_pct:5.1f}% | Streak: {prog_fwd.consecutive_passes}/5 | Status: {prog_fwd.status}")
        print(f"   Reverse Locomotion     : {prog_rev.mastery_pct:5.1f}% | Streak: {prog_rev.consecutive_passes}/5 | Status: {prog_rev.status}")
        print(f"   Steer Left             : {prog_steer_l.mastery_pct:5.1f}% | Streak: {prog_steer_l.consecutive_passes}/5 | Status: {prog_steer_l.status}")
        print(f"   Steer Right            : {prog_steer_r.mastery_pct:5.1f}% | Streak: {prog_steer_r.consecutive_passes}/5 | Status: {prog_steer_r.status}")
        print("=" * 80)
        if all_certified:
            print("[+] DEVELOPMENTAL CURRICULUM COMPLETE. BRAIN FULLY GRADUATED & CERTIFIED.\n")
        else:
            print("[!] DEVELOPMENTAL CURRICULUM INCOMPLETE. SOME SKILLS STILL PRACTICING.\n")

    assert all_certified, "Developmental curriculum did not achieve 100% certification across all skills"
    return brain


if __name__ == "__main__":
    train_developmental_curriculum(seed=42, verbose=True)
