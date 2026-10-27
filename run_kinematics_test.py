"""
Kinematic Solver and Foot Trajectory Generator for Theo Jansen Linkage.
Replicates Figure 1 in the paper.
"""

import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kinematics import solve_single_jansen_leg

def run_kinematics():
    print("=" * 70)
    print("Solving Theo Jansen Linkage Kinematics (Replicating Fig. 1)...")
    print("=" * 70)

    angles = np.linspace(0, 2 * math.pi, 360)
    foot_x = []
    foot_y = []

    for phi in angles:
        joints = solve_single_jansen_leg(phi)
        x, y = joints["Foot"]
        foot_x.append(x)
        foot_y.append(y)

    foot_y = np.array(foot_y)
    ground_thresh = np.min(foot_y) + 0.15 * (np.max(foot_y) - np.min(foot_y))
    stance_mask = foot_y <= ground_thresh
    duty_cycle = np.sum(stance_mask) / len(foot_y) * 100.0

    print(f"Calculated Stance Duty Cycle : {duty_cycle:.1f}% (Nominal ~54%)")
    print(f"Foot Trajectory Height Span  : {np.max(foot_y) - np.min(foot_y):.1f} mm")
    print(f"Foot Trajectory Stride Length: {np.max(foot_x) - np.min(foot_x):.1f} mm")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(foot_x, foot_y, 'b-', linewidth=2, label='Foot Locus Profile')
    ax.axhline(ground_thresh, color='r', linestyle='--', label=f'Ground Stance Threshold ({duty_cycle:.1f}% Duty Cycle)')
    ax.set_xlabel('Horizontal Position X (mm)')
    ax.set_ylabel('Vertical Position Y (mm)')
    ax.set_title('Figure 1: Planar Ovoid Foot Locus of the Theo Jansen Linkage')
    ax.legend(loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_aspect('equal', 'box')
    fig.tight_layout()

    out_path = os.path.join(os.path.dirname(__file__), 'results', 'fig1_jansen_kinematics.png')
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved Figure 1 to: {out_path}\n")

if __name__ == '__main__':
    run_kinematics()
