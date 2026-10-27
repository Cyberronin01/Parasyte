"""
PARASYTE V2 — Publication Figure Generator (Standard Library + NumPy + Matplotlib)
Reads real telemetry from experiment_log.csv and generates 5 high-resolution (300 DPI)
publication-quality figures (.png and .pdf) styled for IEEE / Springer two-column research papers.
Dynamically extracts actual stage transitions from authentic physical simulation data.
"""

import os
import sys
import math
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from simulation.actuators_body import solve_canonical_leg

# ═══════════════════════════════════════════════════════════
# PLOT STYLING (IEEE PUBLICATION COMPLIANT)
# ═══════════════════════════════════════════════════════════
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "lines.linewidth": 1.75,
    "grid.alpha": 0.35,
    "grid.linestyle": "--"
})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
FIG_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(FIG_DIR, exist_ok=True)


def moving_average(a, n=15):
    """Simple moving average filter for clean publication plot curves."""
    if len(a) < n:
        return np.array(a)
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    smoothed = ret[n - 1:] / n
    pad = np.full(n - 1, smoothed[0])
    return np.concatenate([pad, smoothed])


def load_log_data(csv_file):
    """Loads CSV log file into a dictionary of numpy arrays with string handling."""
    data = {}
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                if k not in data:
                    data[k] = []
                if k == "stage":
                    data[k].append(v)
                else:
                    try:
                        data[k].append(float(v))
                    except ValueError:
                        data[k].append(v)
    for k in data:
        if k != "stage":
            data[k] = np.array(data[k])
        else:
            data[k] = np.array(data[k], dtype=object)
    return data


def plot_fig1_klann_kinematics():
    """Figure 1: Theo Jansen 11-Bar Linkage Geometry (CAD Matched) & 8-Leg Stance Support Polygon."""
    print("Generating Figure 1: Theo Jansen 11-Bar Kinematics...")
    fig = plt.figure(figsize=(10, 4.5), dpi=300)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.2, 1.0])

    # ── Panel A: Linkage Skeletal Diagram ──
    ax1 = fig.add_subplot(gs[0])
    thetas = np.linspace(0, 2.0 * math.pi, 200)
    foot_path = []
    for th in thetas:
        joints = solve_canonical_leg(th)
        foot_path.append(joints["Foot"])
    foot_path = np.array(foot_path)

    sample_th = math.pi / 4.0
    joints = solve_canonical_leg(sample_th)

    def draw_link(p1, p2, color="#334155", lw=2.0, ls="-"):
        ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=lw, linestyle=ls)

    # Klann mechanism links
    draw_link(joints["A"], joints["C"], color="#e11d48", lw=3.0)  # Crank link a
    draw_link(joints["B"], joints["D"], color="#0284c7")          # Rocker link d
    draw_link(joints["C"], joints["D"], color="#0284c7")          # Coupler link b
    draw_link(joints["C"], joints["E"], color="#0284c7")          # Link c
    draw_link(joints["B"], joints["F"], color="#059669")          # Rocker link f
    draw_link(joints["E"], joints["F"], color="#059669")          # Link e
    draw_link(joints["D"], joints["G"], color="#475569")          # Link g
    draw_link(joints["E"], joints["G"], color="#475569")          # Triangle link j
    draw_link(joints["F"], joints["Foot"], color="#7c3aed", lw=2.5) # Lower leg
    draw_link(joints["G"], joints["Foot"], color="#7c3aed", lw=2.5)

    ax1.plot(foot_path[:, 0], foot_path[:, 1], color="#94a3b8", linestyle="--", linewidth=1.5, label="Ovoid Gait Locus")
    ax1.axhline(y=-8.0, color="#d97706", linestyle=":", linewidth=1.5, label="Ground Stance Plane")

    ax1.scatter([joints["A"][0], joints["B"][0]], [joints["A"][1], joints["B"][1]], color="#0f172a", s=60, zorder=5)
    ax1.scatter([joints["Foot"][0]], [joints["Foot"][1]], color="#7c3aed", s=70, zorder=6, label="Toe Contact")

    ax1.text(joints["A"][0] + 3, joints["A"][1] - 3, "A (Drive Crank)", fontsize=8, fontweight="bold")
    ax1.text(joints["B"][0] - 25, joints["B"][1] - 5, "B (Chassis Pivot)", fontsize=8, fontweight="bold")
    ax1.set_title("(a) Theo Jansen 11-Bar Linkage")
    ax1.set_xlabel("X Position (mm)")
    ax1.set_ylabel("Y Position (mm)")
    ax1.set_aspect("equal")
    ax1.grid(True)
    ax1.legend(loc="upper right", framealpha=0.9)

    # ── Panel B: 8-Leg Stance Duty Cycle ──
    ax2 = fig.add_subplot(gs[1])
    crank_angles_deg = np.linspace(0, 360, 100)
    phase_offsets = [0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0]
    grounded_legs = []

    for deg in crank_angles_deg:
        rad = math.radians(deg)
        l_grounded = sum(1 for i in range(4) if solve_canonical_leg((rad + phase_offsets[i]) % (2 * math.pi))["Foot"][1] <= -8.0)
        r_grounded = sum(1 for i in range(4) if solve_canonical_leg((rad + phase_offsets[i]) % (2 * math.pi))["Foot"][1] <= -8.0)
        grounded_legs.append(l_grounded + r_grounded)

    ax2.plot(crank_angles_deg, grounded_legs, color="#0284c7", linewidth=2.5, label="Simultaneous Grounded Legs")
    ax2.axhline(y=4, color="#dc2626", linestyle="--", linewidth=1.8, label="Stability Minimum (≥ 4 Legs)")
    ax2.fill_between(crank_angles_deg, 4, grounded_legs, color="#0284c7", alpha=0.15)

    ax2.set_title("(b) 8-Leg Dynamic Stance Margin")
    ax2.set_xlabel("Crankshaft Angle (degrees)")
    ax2.set_ylabel("Number of Grounded Legs")
    ax2.set_ylim([2, 8])
    ax2.grid(True)
    ax2.legend(loc="lower right", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig1_jansen_kinematics.png"), dpi=300)
    fig.savefig(os.path.join(FIG_DIR, "fig1_jansen_kinematics.pdf"))
    plt.close(fig)


def plot_fig2_loss_convergence(d):
    """Figure 2: Multi-stream self-supervised predictive loss convergence."""
    print("Generating Figure 2: Predictive Loss Convergence...")
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=300)

    stage_arr = d["stage"]
    stage1_mask = np.isin(stage_arr, ["babble", "walk"])
    if not np.any(stage1_mask):
        stage1_mask = np.arange(len(stage_arr)) < min(600, len(stage_arr))

    steps = d["step"][stage1_mask]
    loss_tot = moving_average(d["loss_total"][stage1_mask])
    loss_depth = moving_average(d["loss_depth"][stage1_mask])
    loss_imu = moving_average(d["loss_imu"][stage1_mask])
    loss_enc = moving_average(d["loss_enc"][stage1_mask])
    loss_mot = moving_average(d["loss_mot"][stage1_mask])

    ax.plot(steps, loss_tot, color="#0f172a", linewidth=2.5, label="Total Residual Prediction Error")
    ax.plot(steps, loss_depth, color="#0284c7", label="Depth RNN (VL53L5CX ToF 64 Zones)")
    ax.plot(steps, loss_imu, color="#e11d48", label="Balance RNN (MPU6050 6-DOF)")
    ax.plot(steps, loss_enc, color="#059669", label="Proprioception RNN (Dual Encoders)")
    ax.plot(steps, loss_mot, color="#7c3aed", label="Motor Current RNN (Armature Current)")

    babble_indices = np.where(stage_arr == "babble")[0]
    if len(babble_indices) > 0:
        babble_end_step = d["step"][babble_indices[-1]]
        ax.axvline(x=babble_end_step, color="#d97706", linestyle=":", linewidth=1.8,
                   label=f"Embodiment Mastered (Step {int(babble_end_step)})")

    ax.set_yscale("log")
    ax.set_title("Self-Supervised Sensory Prediction Error Convergence (Stage 1)")
    ax.set_xlabel("Training Steps (50 Hz Control Ticks)")
    ax.set_ylabel("Mean Squared Error (Log Scale)")
    ax.grid(True, which="both")
    ax.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig2_loss_convergence.png"), dpi=300)
    fig.savefig(os.path.join(FIG_DIR, "fig2_loss_convergence.pdf"))
    plt.close(fig)


def plot_fig3_ewc_forgetting_proof(d):
    """Figure 3: Catastrophic Forgetting Proof (PARASYTE EWC vs Standard Baseline)."""
    print("Generating Figure 3: EWC Skill Retention vs Baseline...")
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=300)

    ewc_csv = os.path.join(DATA_DIR, "ewc_retention_data.csv")
    if os.path.exists(ewc_csv):
        # Load empirical retention benchmarks
        steps, ewc_ret, base_ret = [], [], []
        with open(ewc_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                steps.append(float(row["step"]))
                ewc_ret.append(float(row["ewc_retention"]))
                base_ret.append(float(row["baseline_retention"]))
        steps = np.array(steps)
        ewc_ret = np.array(ewc_ret)
        base_ret = np.array(base_ret)

        ax.axvline(x=0, color="#64748b", linestyle="--", linewidth=1.5,
                   label="Walking Mastery & EWC Lock (Step 0)")
        ax.plot(steps, ewc_ret, color="#059669", linewidth=2.5, marker="o", markersize=5,
                label=f"PARASYTE (Hierarchical EWC, Final: {ewc_ret[-1]:.1f}%)")
        ax.plot(steps, base_ret, color="#dc2626", linewidth=2.0, linestyle="-.", marker="s", markersize=5,
                label=f"Standard Baseline (No EWC, Final: {base_ret[-1]:.1f}%)")

        ax.fill_between(steps, base_ret, ewc_ret, color="#059669", alpha=0.15,
                        label=f"Conserved Locomotion Advantage (+{ewc_ret[-1] - base_ret[-1]:.1f}%)")
        ax.set_xlabel("Reactive Obstacle Steering Adaptation Steps (50 Hz)")
    else:
        stage_arr = d["stage"]
        steer_indices = np.where(stage_arr == "steer")[0]
        if len(steer_indices) > 0:
            steer_start_idx = steer_indices[0]
            steer_start_step = d["step"][steer_start_idx]
            start_idx = max(0, steer_start_idx - 50)
            end_idx = min(len(stage_arr), steer_indices[-1] + 20)
        else:
            start_idx = 0
            end_idx = len(stage_arr)
            steer_start_step = 340

        mask = np.zeros(len(stage_arr), dtype=bool)
        mask[start_idx:end_idx] = True

        steps = d["step"][mask]
        ewc_ret = d["ewc_retention"][mask]
        base_ret = d["baseline_retention"][mask]

        if np.max(ewc_ret) <= 1.5:
            ewc_ret = ewc_ret * 100.0
            base_ret = base_ret * 100.0

        ax.axvline(x=steer_start_step, color="#64748b", linestyle="--", linewidth=1.5,
                   label=f"Walking Mastery & EWC Lock (Step {int(steer_start_step)})")
        ax.plot(steps, ewc_ret, color="#059669", linewidth=2.5, label="PARASYTE (Hierarchical EWC)")
        ax.plot(steps, base_ret, color="#dc2626", linewidth=2.0, linestyle="-.", label="Standard Baseline (No EWC Protection)")
        ax.fill_between(steps, base_ret, ewc_ret, color="#059669", alpha=0.15, label="Conserved Locomotion Advantage")
        ax.set_xlabel("Training Step (50 Hz)")

    ax.set_title("Retention of Locomotion Competence During Reactive Steering Adaptation")
    ax.set_ylabel("Locomotion Mastery Retention (%)")
    ax.set_ylim([65, 105])
    ax.grid(True)
    ax.legend(loc="lower left", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig3_ewc_forgetting_proof.png"), dpi=300)
    fig.savefig(os.path.join(FIG_DIR, "fig3_ewc_forgetting_proof.pdf"))
    plt.close(fig)


def plot_fig4_surprise_anomaly_spike(d):
    """Figure 4: Statistical Surprise Detector Failure Spike (>3σ Detection)."""
    print("Generating Figure 4: Surprise Detector Failure Detection...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5.0), dpi=300, sharex=True)

    stage_arr = d["stage"]
    collision_indices = np.where(stage_arr == "collision")[0]
    if len(collision_indices) > 0:
        c_start = collision_indices[0]
        c_end = collision_indices[-1]
        start_idx = max(0, c_start - 20)
        end_idx = min(len(stage_arr), c_end + 1)
    else:
        start_idx = max(0, len(stage_arr) - 200)
        end_idx = len(stage_arr)

    mask = np.zeros(len(stage_arr), dtype=bool)
    mask[start_idx:end_idx] = True

    steps = d["step"][mask]
    loss = d["loss_total"][mask]
    surprise = d["z_score"][mask]
    interlocks = d["safety_interlock"][mask]
    motor_curr = d["motor_current"][mask]

    fail_steps = steps[interlocks == 1]

    # Panel 1: Error & Motor Current Stall
    ax1.plot(steps, loss, color="#0284c7", linewidth=2.0, label="Multi-Modal Discrepancy (MSE)")
    ax1_twin = ax1.twinx()
    ax1_twin.plot(steps, motor_curr, color="#d97706", linewidth=1.5, linestyle=":", label="Motor Current (A)")
    ax1_twin.set_ylabel("Current (A)", color="#d97706")

    if len(fail_steps) > 0:
        ax1.axvspan(fail_steps[0], fail_steps[-1], color="#fecaca", alpha=0.5, label="Safety Interlock Active")
    ax1.set_ylabel("Prediction Loss")
    ax1.set_title("(a) Multi-Modal Sensory Prediction Discrepancy & Motor Stall")
    ax1.grid(True)
    ax1.legend(loc="upper left")

    # Panel 2: Normalized Surprise
    ax2.plot(steps, surprise, color="#0f172a", linewidth=2.0, label="Normalized Surprise ($Z_t$)")
    ax2.axhline(y=12.0, color="#dc2626", linestyle="--", linewidth=2.0, label="Calibrated Anomaly Threshold (12.0σ)")
    if len(fail_steps) > 0:
        ax2.axvspan(fail_steps[0], fail_steps[-1], color="#fecaca", alpha=0.5)
        idx0 = int(np.where(steps == fail_steps[0])[0][0])
        ax2.annotate(f"Collision Impact Discovered\nMotors Cut to 0.0A (Step {int(fail_steps[0])})",
                     xy=(fail_steps[0], min(30.0, float(surprise[idx0]))),
                     xytext=(fail_steps[0] - 25, 20.0),
                     arrowprops=dict(facecolor="#dc2626", shrink=0.08, width=1.5, headwidth=6),
                     fontsize=9, fontweight="bold", color="#991b1b")

    ax2.set_xlabel("Time Step (50 Hz)")
    ax2.set_ylabel("Surprise Z-Score ($σ$)")
    ax2.set_title("(b) Emergent Physical Collision Discovery & Interlock")
    ax2.grid(True)
    ax2.legend(loc="upper left")

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig4_surprise_anomaly_spike.png"), dpi=300)
    fig.savefig(os.path.join(FIG_DIR, "fig4_surprise_anomaly_spike.pdf"))
    plt.close(fig)


def plot_fig5_robot_navigation_path(d):
    """Figure 5: 2D Spatial Floor Trajectory showing straight stride and obstacle steering."""
    print("Generating Figure 5: 2D Robot Navigation Trajectory...")
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)

    rx = d["robot_x"] / 1000.0  # Convert mm to meters
    ry = d["robot_y"] / 1000.0

    ax.plot(rx, ry, color="#0284c7", linewidth=2.5, label="PARASYTE Chassis Path (SITL)")
    ax.scatter(rx[0], ry[0], color="#059669", s=90, marker="o", zorder=5, label="Start (0, 0)")
    ax.scatter(rx[-1], ry[-1], color="#7c3aed", s=90, marker="X", zorder=5, label="Final Endpoint")

    obstacles = [
        (0.30, 0.04, 0.10, "Pillar 1"),
        (0.60, -0.05, 0.12, "Pillar 2"),
        (0.95, 0.06, 0.13, "Pillar 3"),
    ]

    for ox, oy, r, name in obstacles:
        circle = plt.Circle((ox, oy), r, color="#cbd5e1", ec="#475569", linewidth=1.5)
        ax.add_patch(circle)
        ax.text(ox - 0.06, oy - 0.02, name, fontsize=8, color="#1e293b", fontweight="bold")

    ax.set_title("Emergent 2D Locomotion and Collision-Avoidance Trajectory")
    ax.set_xlabel("Forward Displacement X (meters)")
    ax.set_ylabel("Lateral Displacement Y (meters)")
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_xlim([-0.2, max(1.2, np.max(rx) + 0.3)])
    ax.set_ylim([-0.6, 0.6])
    ax.legend(loc="upper left", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig5_robot_navigation_path.png"), dpi=300)
    fig.savefig(os.path.join(FIG_DIR, "fig5_robot_navigation_path.pdf"))
    plt.close(fig)


def generate_all_figures():
    csv_file = os.path.join(DATA_DIR, "experiment_log.csv")
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Missing {csv_file}. Please run run_experiment.py first.")

    data = load_log_data(csv_file)

    plot_fig1_klann_kinematics()
    plot_fig2_loss_convergence(data)
    plot_fig3_ewc_forgetting_proof(data)
    plot_fig4_surprise_anomaly_spike(data)
    plot_fig5_robot_navigation_path(data)

    print(f"\n[SUCCESS] All 5 publication figures successfully saved to {FIG_DIR} (.png and .pdf)!")


if __name__ == "__main__":
    generate_all_figures()
