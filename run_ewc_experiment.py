"""
Generates authentic step-by-step EWC retention trajectory data during active obstacle steering adaptation.
Saves results to data/ewc_retention_data.csv for Figure 3 plotting.
"""

import os
import sys
import math
import csv
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from brain import ParasyteBrain
from simulation.actuators_body import ArachnobeestBody
from simulation.sensor_suite import ArenaWorld, SimulatedObstacle, HALBridge
from simulation.tof_steering import compute_reactive_tof_steering

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(DATA_DIR, exist_ok=True)
CSV_OUT = os.path.join(DATA_DIR, "ewc_retention_data.csv")

def main():
    print("Generating authentic EWC vs Baseline retention trajectory...")
    torch.manual_seed(42)
    np.random.seed(42)

    brain = ParasyteBrain()

    # 1. Master Task A (Forward Locomotion)
    arena_open = ArenaWorld()
    arena_open.obstacles = []
    body_walk = ArachnobeestBody(dt=0.02)
    hal_walk = HALBridge(body_walk, arena_open)

    task_a_dataset = []
    pkt_prev = hal_walk.step(0.65, 0.65)
    prev_tensors = {k: torch.from_numpy(pkt_prev[k if k != 'motor' else 'motor_echo']) for k in ['depth', 'depth_front', 'depth_rear', 'imu', 'encoders', 'motor']}

    for step in range(1, 101):
        pkt_curr = hal_walk.step(0.65, 0.65)
        curr_tensors = {k: torch.from_numpy(pkt_curr[k if k != 'motor' else 'motor_echo']) for k in ['depth', 'depth_front', 'depth_rear', 'imu', 'encoders', 'motor']}

        emb_prev, preds_curr = brain.perception(prev_tensors['depth'], prev_tensors['imu'],
                                                prev_tensors['encoders'], prev_tensors['motor'], prev_tensors.get('depth_rear'))
        st_prev, _ = brain.fusion(emb_prev)
        act_prev = brain.locomotion(st_prev)

        loss_walk = torch.mean((act_prev - 0.65) ** 2)
        loss_perc = sum(torch.mean((preds_curr[k] - curr_tensors[k]) ** 2) for k in preds_curr if k in curr_tensors)
        wm_pred = brain.world_model(st_prev, act_prev, prev_tensors['depth'])
        loss_wm = torch.mean((wm_pred["depth_64"] - curr_tensors['depth']) ** 2) + torch.mean((wm_pred["yaw_rate"] - curr_tensors['imu'][5]) ** 2)
        total_loss = loss_walk + loss_perc + 0.5 * loss_wm

        brain.opt_perc.zero_grad()
        brain.opt_fusion.zero_grad()
        brain.opt_world_model.zero_grad()
        brain.opt_locomotion.zero_grad()
        total_loss.backward()
        brain.opt_perc.step()
        brain.opt_fusion.step()
        brain.opt_world_model.step()
        brain.opt_locomotion.step()

        task_a_dataset.append(prev_tensors)
        prev_tensors = curr_tensors

    def evaluate_straight_track(eval_brain):
        eval_body = ArachnobeestBody(dt=0.02)
        eval_hal = HALBridge(eval_body, arena_open)
        eval_pkt = eval_hal.step(0.65, 0.65)

        for _ in range(80):
            t_d, t_i, t_e, t_m = [torch.from_numpy(eval_pkt[k]) for k in ['depth', 'imu', 'encoders', 'motor_echo']]
            with torch.no_grad():
                emb, _ = eval_brain.perception(t_d, t_i, t_e, t_m)
                st, _ = eval_brain.fusion(emb)
                act = eval_brain.locomotion(st)
            eval_pkt = eval_hal.step(float(act[0].item()), float(act[1].item()))

        gt = eval_body.get_ground_truth()
        dist_x = float(gt["pose"][0])
        drift_deg = abs(math.degrees(float(gt["pose"][2])))
        return dist_x, drift_deg

    dist_before, drift_before = evaluate_straight_track(brain)
    print(f"Task A Mastered Baseline: Forward = {dist_before:.2f} mm | Drift = {drift_before:.2f} deg")

    # 2. Compute Fisher Information
    def criterion_fn(sample):
        emb, preds = brain.perception(sample['depth'], sample['imu'], sample['encoders'], sample['motor'])
        st, _ = brain.fusion(emb)
        act = brain.locomotion(st)
        wm_pred = brain.world_model(st, act, sample['depth'])
        loss_walk = torch.mean((act - 0.65) ** 2)
        loss_preds = sum(torch.mean((preds[k] - sample[k]) ** 2) for k in ['imu', 'encoders', 'motor'])
        loss_wm = torch.mean((wm_pred["clearance"] - 1.0) ** 2) + torch.mean((wm_pred["yaw_rate"] - 0.0) ** 2)
        return loss_walk + 0.1 * loss_preds + 0.1 * loss_wm

    brain.ewc.compute_fisher(task_a_dataset[-40:], criterion_fn)

    # Clone baseline
    brain_no_ewc = ParasyteBrain()
    brain_no_ewc.perception.load_state_dict(brain.perception.state_dict())
    brain_no_ewc.fusion.load_state_dict(brain.fusion.state_dict())
    brain_no_ewc.world_model.load_state_dict(brain.world_model.state_dict())
    brain_no_ewc.locomotion.load_state_dict(brain.locomotion.state_dict())
    brain_no_ewc.steering.load_state_dict(brain.steering.state_dict())
    brain_no_ewc.ewc.active = False

    # 3. Training Stage 2 with periodic retention evaluation
    arena_obs = ArenaWorld()
    arena_obs.obstacles = [
        SimulatedObstacle(120.0, 30.0, 80.0, "Pillar Near Left"),
        SimulatedObstacle(240.0, -35.0, 80.0, "Pillar Near Right"),
        SimulatedObstacle(360.0, 40.0, 90.0, "Pillar Far Left"),
    ]
    hal_obs_ewc = HALBridge(ArachnobeestBody(dt=0.02), arena_obs)
    hal_obs_no_ewc = HALBridge(ArachnobeestBody(dt=0.02), arena_obs)

    brain.ewc.importance = 500.0
    pwm_tl_e, pwm_tr_e = 0.65, 0.65
    pwm_tl_ne, pwm_tr_ne = 0.65, 0.65

    records = []
    # Log initial point at step 0
    records.append({
        "step": 0,
        "ewc_retention": 100.0,
        "ewc_drift": drift_before,
        "baseline_retention": 100.0,
        "baseline_drift": drift_before
    })

    total_steer_steps = 80
    eval_interval = 10

    for step in range(1, total_steer_steps + 1):
        # EWC update
        pkt_e = hal_obs_ewc.step(pwm_tl_e, pwm_tr_e)
        t_d, t_i, t_e, t_m = [torch.from_numpy(pkt_e[k]) for k in ['depth', 'imu', 'encoders', 'motor_echo']]
        emb_e, _ = brain.perception(t_d, t_i, t_e, t_m)
        st_e, _ = brain.fusion(emb_e)
        base_act = brain.locomotion(st_e)
        steer_act = brain.steering(st_e, base_act)

        pwm_tl_e, pwm_tr_e, _ = compute_reactive_tof_steering(pkt_e["depth"])
        steer_target_e = torch.tensor([pwm_tl_e, pwm_tr_e], dtype=torch.float32)

        loss_steer_e = torch.mean((steer_act - steer_target_e) ** 2)
        penalty = brain.ewc.penalty()
        total_loss_ewc = loss_steer_e + penalty

        brain.opt_steering.zero_grad()
        brain.opt_locomotion.zero_grad()
        brain.opt_fusion.zero_grad()
        total_loss_ewc.backward()
        brain.opt_steering.step()
        brain.opt_locomotion.step()
        brain.opt_fusion.step()

        # Baseline update
        pkt_ne = hal_obs_no_ewc.step(pwm_tl_ne, pwm_tr_ne)
        t_d, t_i, t_e, t_m = [torch.from_numpy(pkt_ne[k]) for k in ['depth', 'imu', 'encoders', 'motor_echo']]
        emb_ne, _ = brain_no_ewc.perception(t_d, t_i, t_e, t_m)
        st_ne, _ = brain_no_ewc.fusion(emb_ne)
        base_act_ne = brain_no_ewc.locomotion(st_ne)
        steer_act_ne = brain_no_ewc.steering(st_ne, base_act_ne)

        pwm_tl_ne, pwm_tr_ne, _ = compute_reactive_tof_steering(pkt_ne["depth"])
        steer_target_ne = torch.tensor([pwm_tl_ne, pwm_tr_ne], dtype=torch.float32)

        loss_no_ewc = torch.mean((steer_act_ne - steer_target_ne) ** 2)
        brain_no_ewc.opt_steering.zero_grad()
        brain_no_ewc.opt_locomotion.zero_grad()
        brain_no_ewc.opt_fusion.zero_grad()
        loss_no_ewc.backward()
        brain_no_ewc.opt_steering.step()
        brain_no_ewc.opt_locomotion.step()
        brain_no_ewc.opt_fusion.step()

        # Evaluate at intervals
        if step % eval_interval == 0:
            d_ewc, dr_ewc = evaluate_straight_track(brain)
            d_base, dr_base = evaluate_straight_track(brain_no_ewc)
            ret_ewc = (d_ewc / dist_before) * 100.0
            ret_base = (d_base / dist_before) * 100.0
            print(f"  Step {step:2d} | EWC Retention: {ret_ewc:5.1f}% (Drift: {dr_ewc:.2f} deg) | Baseline Retention: {ret_base:5.1f}% (Drift: {dr_base:.2f} deg)")
            records.append({
                "step": step,
                "ewc_retention": ret_ewc,
                "ewc_drift": dr_ewc,
                "baseline_retention": ret_base,
                "baseline_drift": dr_base
            })

    with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "ewc_retention", "ewc_drift", "baseline_retention", "baseline_drift"])
        writer.writeheader()
        writer.writerows(records)

    print(f"Saved EWC retention trajectory to {CSV_OUT}")

    # Plot Figure 3
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [r["step"] for r in records]
    b_drifts = [r["baseline_drift"] for r in records]
    e_drifts = [r["ewc_drift"] for r in records]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, b_drifts, 'r--o', label='Baseline (Unregularized Fine-Tuning)')
    ax.plot(steps, e_drifts, 'g-s', label=r'PARASYTE (EWC Anchored, $\lambda=500$)')
    ax.set_xlabel('Task 2 Adaptation Steps')
    ax.set_ylabel('Straight-Line Heading Drift (deg)')
    ax.set_title('Figure 3: Retention of Straight-Line Locomotion During Steering Adaptation')
    ax.legend(frameon=True)
    ax.grid(True, linestyle=':', alpha=0.6)
    fig.tight_layout()

    fig_out = os.path.join(DATA_DIR, "fig3_ewc_forgetting_proof.png")
    fig.savefig(fig_out, dpi=300)
    plt.close(fig)
    print(f"Saved Figure 3 plot to: {fig_out}")


if __name__ == "__main__":
    main()
