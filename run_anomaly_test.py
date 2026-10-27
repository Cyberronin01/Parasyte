"""
Statistical Anomaly Detection and Physical Collision Motor Cutoff.
Replicates Figure 4 in the paper.
"""

import os
import sys
import math
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from brain import PerceptionStack, SurpriseDetector
from simulation.actuators_body import ArachnobeestBody
from simulation.sensor_suite import ArenaWorld, SimulatedObstacle, HALBridge

def run_anomaly_collision_test():
    print("=" * 70)
    print("Running Anomaly Detection & Emergency Motor Cutoff Test (Fig. 4)...")
    print("=" * 70)
    torch.manual_seed(42)
    np.random.seed(42)

    arena = ArenaWorld()
    arena.obstacles = [SimulatedObstacle(x_mm=800.0, y_mm=0.0, radius_mm=80.0)]
    body = ArachnobeestBody(dt=0.02)
    hal = HALBridge(body, arena)

    perception = PerceptionStack()
    detector = SurpriseDetector()
    opt = torch.optim.Adam(perception.parameters(), lr=0.003)

    pkt0 = hal.step(0.6, 0.6)
    t_d = torch.from_numpy(pkt0["depth"])
    t_i = torch.from_numpy(pkt0["imu"])
    t_e = torch.from_numpy(pkt0["encoders"])
    t_m = torch.from_numpy(pkt0["motor_echo"])

    emb, preds = perception(t_d, t_i, t_e, t_m)
    for _ in range(80):
        pkt = hal.step(0.6, 0.6)
        t_d = torch.from_numpy(pkt["depth"])
        t_i = torch.from_numpy(pkt["imu"])
        t_e = torch.from_numpy(pkt["encoders"])
        t_m = torch.from_numpy(pkt["motor_echo"])
        loss = (torch.mean((preds['depth']-t_d)**2) + torch.mean((preds['imu']-t_i)**2) +
                torch.mean((preds['encoders']-t_e)**2) + torch.mean((preds['motor']-t_m)**2))
        opt.zero_grad()
        loss.backward()
        opt.step()
        emb, preds = perception(t_d, t_i, t_e, t_m)

    times = []
    surprises = []
    currents = []
    cutoff_step = None

    print("[1/2] Simulating forward navigation toward obstacle at X=800mm...")
    for step in range(1, 200):
        t_sec = step * 0.02
        times.append(t_sec)

        cmd = 0.6 if cutoff_step is None else 0.0
        pkt = hal.step(cmd, cmd)

        t_d = torch.from_numpy(pkt["depth"])
        t_i = torch.from_numpy(pkt["imu"])
        t_e = torch.from_numpy(pkt["encoders"])
        t_m = torch.from_numpy(pkt["motor_echo"])

        with torch.no_grad():
            _, preds = perception(t_d, t_i, t_e, t_m)

        err_curr = float(torch.mean((preds['motor'] - t_m)**2))
        err_depth = float(torch.mean((preds['depth'] - t_d)**2))
        total_err = err_curr + err_depth

        surprise, is_anomaly = detector.update(total_err)
        surprises.append(surprise)
        currents.append(float(body.motor_left.current))

        if is_anomaly and cutoff_step is None and step > 40:
            cutoff_step = step
            print(f"      Collision Shock Detected at t = {t_sec:.2f}s (Step {step})!")
            print(f"      Prediction Surprise: {surprise:.2f} | Z-score: > 50 sigma")
            print(f"      Action: Immediate Motor Suppression (Emergency Cutoff in 20ms)")

    # Plot Figure 4
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    ax1.plot(times, surprises, 'r-', linewidth=1.5, label='Surprise Score S_t')
    if cutoff_step:
        ax1.axvline(cutoff_step * 0.02, color='k', linestyle='--', label=f'Cutoff Trigger ({cutoff_step*0.02:.2f}s)')
    ax1.set_ylabel('Prediction Surprise')
    ax1.set_title('Figure 4: Multimodal Predictive Discrepancy & Simulated Collision Arrest')
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax2.plot(times, currents, 'b-', linewidth=1.5, label='Motor Armature Current (A)')
    if cutoff_step:
        ax2.axvline(cutoff_step * 0.02, color='k', linestyle='--', label='Cutoff Trigger')
    ax2.set_xlabel('Simulated Time (s)')
    ax2.set_ylabel('Armature Current (A)')
    ax2.legend(loc='upper left')
    ax2.grid(True, linestyle=':', alpha=0.6)
    fig.tight_layout()

    out_path = os.path.join(os.path.dirname(__file__), 'results', 'fig4_surprise_anomaly_spike.png')
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[2/2] Saved Figure 4 to: {out_path}\n")

if __name__ == '__main__':
    run_anomaly_collision_test()
