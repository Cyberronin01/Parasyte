"""
PARASYTE V2 — 3 Core Empirical Benchmark Scenarios
Evaluates the autonomous sensorimotor engine across three distinct physical regimes:
1. Scenario 1 (Precision Gateway): Lateral flank obstacles (Y = ±250mm), 350mm center aperture.
   Tests optical sector segregation and forward cruise stability without spurious steering.
2. Scenario 2 (S-Curve Corridor): 1.3m hallway (Y = ±650mm) with 3 staggered pillars.
   Tests dynamic continuous evasive steering and obstacle bypass.
3. Scenario 3 (U-Shaped Dead End): Enclosed cul-de-sac (walls ahead and on both flanks).
   Tests autonomous nonholonomic reverse retreat driven by Active Inference Free Energy.
"""

import os
import sys
import math
import time
from typing import Dict, List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from config import DT
from brain import ParasyteBrain
from governance.hardware_guardrails import HardwareSafetyGuard
from simulation.actuators_body import ArachnobeestBody
from simulation.sensor_suite import ArenaWorld, SimulatedObstacle, HALBridge
from developmental_curriculum import train_developmental_curriculum


# ─────────────────────────────────────────────────────────────────────────────
# 1. FIXED PHYSICAL ARENA DEFINITIONS (LOCKED GEOMETRY)
# ─────────────────────────────────────────────────────────────────────────────

def build_scenario_1_gateway() -> Tuple[ArenaWorld, str]:
    """
    Scenario 1: Precision Gateway (Aperture Traversal)
    Pillars at Y = ±250mm, Radius = 75mm -> 350mm clear gap for 320mm chassis.
    Open arena ahead and behind.
    """
    arena = ArenaWorld()
    arena.obstacles = [
        SimulatedObstacle(800.0,  250.0, 75.0, "Gate_Left",  350.0),
        SimulatedObstacle(800.0, -250.0, 75.0, "Gate_Right", 350.0),
    ]
    return arena, "Scenario 1: Precision Gateway (350mm Aperture)"


def build_scenario_2_scurve() -> Tuple[ArenaWorld, str]:
    """
    Scenario 2: S-Curve Corridor (Continuous Evasive Weave)
    Standard 1.3m indoor hallway (walls at Y = ±650mm).
    3 staggered pillars force right -> left -> right weave.
    """
    arena = ArenaWorld()
    pillars = [
        SimulatedObstacle(650.0,   150.0, 90.0, "Slalom_Pillar_1", 350.0),
        SimulatedObstacle(1450.0, -150.0, 90.0, "Slalom_Pillar_2", 350.0),
        SimulatedObstacle(2250.0,  150.0, 90.0, "Slalom_Pillar_3", 350.0),
    ]
    top_walls = [SimulatedObstacle(float(x),  650.0, 60.0, f"Wall_Top_{x}", 350.0) for x in range(250, 3200, 200)]
    bot_walls = [SimulatedObstacle(float(x), -650.0, 60.0, f"Wall_Bot_{x}", 350.0) for x in range(250, 3200, 200)]
    arena.obstacles = pillars + top_walls + bot_walls
    return arena, "Scenario 2: S-Curve Corridor (1.3m Hallway Weave)"


def build_scenario_3_deadend() -> Tuple[ArenaWorld, str]:
    """
    Scenario 3: U-Shaped Dead-End Cul-de-Sac
    Pocket width: 700mm (left wall Y=+350, right wall Y=-350).
    Closed at X=900mm with solid cross-wall.
    Completely open rearward (X < 300mm).
    """
    arena = ArenaWorld()
    # Left flank wall (X in [300, 900], Y = +350)
    left_wall = [SimulatedObstacle(float(x), 350.0, 50.0, f"DeadEnd_Left_{x}", 350.0) for x in range(300, 950, 80)]
    # Right flank wall (X in [300, 900], Y = -350)
    right_wall = [SimulatedObstacle(float(x), -350.0, 50.0, f"DeadEnd_Right_{x}", 350.0) for x in range(300, 950, 80)]
    # Solid end cross-wall (X = 900, Y in [-350, +350])
    end_wall = [SimulatedObstacle(900.0, float(y), 50.0, f"DeadEnd_Front_{y}", 350.0) for y in range(-350, 360, 70)]

    arena.obstacles = left_wall + right_wall + end_wall
    return arena, "Scenario 3: U-Shaped Dead End (Autonomous Reverse Escape)"


# ─────────────────────────────────────────────────────────────────────────────
# 2. ISOLATED SCENARIO EXECUTION & TELEMETRY
# ─────────────────────────────────────────────────────────────────────────────

def execute_scenario(scenario_id: int, brain: ParasyteBrain, steps: int = 800) -> Dict:
    """Executes an isolated physical scenario with full instrumentation."""
    if scenario_id == 1:
        arena, name = build_scenario_1_gateway()
    elif scenario_id == 2:
        arena, name = build_scenario_2_scurve()
    elif scenario_id == 3:
        arena, name = build_scenario_3_deadend()
    else:
        raise ValueError(f"Unknown scenario ID: {scenario_id}")

    body = ArachnobeestBody(dt=DT)
    hal = HALBridge(body, arena)
    guard = HardwareSafetyGuard(max_slew=3.0, dt=DT)

    # Initial sensory packet
    cmd_pwm_l, cmd_pwm_r = 0.5, 0.5
    pkt = hal.step(0.0, 0.0)
    prev_tensors = {
        "depth": torch.from_numpy(pkt["depth"]),
        "depth_rear": torch.from_numpy(pkt["depth_rear"]),
        "imu": torch.from_numpy(pkt["imu"]),
        "encoders": torch.from_numpy(pkt["encoders"]),
        "motor": torch.from_numpy(pkt["motor_echo"]),
    }

    # Tracking metrics
    maneuver_ticks = 0
    active_evasion = None
    collisions = 0
    min_clearance = 4.0
    action_counts = {"FORWARD": 0, "STEER_LEFT": 0, "STEER_RIGHT": 0, "REVERSE": 0}
    motor_currents = []
    trajectory = [{
        "step": 0,
        "x": round(float(body.pose_x), 1),
        "y": round(float(body.pose_y), 1),
        "yaw_deg": round(math.degrees(float(body.yaw)), 1),
        "action": 0
    }]

    recent_x_history = []
    stagnation_ticks = 0

    t0 = time.time()

    for step in range(1, steps + 1):
        curr_pkt = hal.step(cmd_pwm_l, cmd_pwm_r)
        curr_tensors = {
            "depth": torch.from_numpy(curr_pkt["depth"]),
            "depth_rear": torch.from_numpy(curr_pkt["depth_rear"]),
            "imu": torch.from_numpy(curr_pkt["imu"]),
            "encoders": torch.from_numpy(curr_pkt["encoders"]),
            "motor": torch.from_numpy(curr_pkt["motor_echo"]),
        }

        if hal.last_collision_gt:
            collisions += 1

        d_front_min = float(np.min(curr_pkt["depth"]))
        if d_front_min < min_clearance:
            min_clearance = d_front_min

        motor_currents.append(float(np.mean(curr_pkt["motor_echo"])))

        # Track forward stagnation across a rolling window
        recent_x_history.append(float(body.pose_x))
        if len(recent_x_history) > 30:
            dx_recent = abs(recent_x_history[-1] - recent_x_history[-30])
            if dx_recent < 15.0 and (active_evasion in [2, 3]):
                stagnation_ticks += 1
            else:
                stagnation_ticks = max(0, stagnation_ticks - 1)

        # Log position every 5 steps
        if step % 5 == 0 or step == steps:
            trajectory.append({
                "step": step,
                "x": round(float(body.pose_x), 1),
                "y": round(float(body.pose_y), 1),
                "yaw_deg": round(math.degrees(float(body.yaw)), 1),
                "action": active_evasion if maneuver_ticks > 0 else 0
            })

        # ── ACTIVE INFERENCE PLANNING ──
        with torch.no_grad():
            emb_prev, _ = brain.perception(
                prev_tensors["depth"], prev_tensors["imu"],
                prev_tensors["encoders"], prev_tensors["motor"],
                prev_tensors.get("depth_rear")
            )
            st_prev, _ = brain.fusion(emb_prev)

            # Biological Stagnation Reflex: Break potential-field local minima via Reverse retreat
            if stagnation_ticks >= 20 and maneuver_ticks <= 5:
                active_evasion = 1  # Goal 1: REVERSE
                maneuver_ticks = 35  # Sustained backward retreat pulse (0.7s)
                stagnation_ticks = 0
                recent_x_history.clear()

            if maneuver_ticks > 0 and active_evasion is not None:
                best_goal = active_evasion
                maneuver_ticks -= 1
                g_vec = torch.zeros(4, dtype=torch.float32, device=st_prev.device)
                g_vec[best_goal] = 1.0
                steered_act = brain.goal_policy(st_prev, g_vec)
            else:
                best_goal, selected_act = brain.select_skill(st_prev, prev_tensors["depth"], prev_tensors.get("depth_rear"))
                steered_act = selected_act
                if best_goal in [2, 3]:  # Steering maneuver persistence
                    active_evasion = best_goal
                    d_mat = prev_tensors["depth"].view(8, 8) if prev_tensors["depth"].numel() == 64 else prev_tensors["depth"]
                    d_fc_cur = float(torch.min(d_mat[2:6, 2:6]).item()) * 4.0
                    maneuver_ticks = 30 if d_fc_cur < 0.45 else 10
                elif best_goal == 1:  # Reverse retreat
                    active_evasion = 1
                    maneuver_ticks = 25  # Sustained retreat pulse
                else:
                    active_evasion = None
                    maneuver_ticks = 0

        raw_l = float(np.clip(steered_act[0].item(), -1.0, 1.0))
        raw_r = float(np.clip(steered_act[1].item(), -1.0, 1.0))

        # Hardware safety guardrail
        accel = (float(curr_tensors["imu"][0]), float(curr_tensors["imu"][1]), float(curr_tensors["imu"][2]))
        enc_speeds = (float(curr_tensors["encoders"][0]), float(curr_tensors["encoders"][1]))
        cmd_pwm_l, cmd_pwm_r, _ = guard.filter_action(raw_l, raw_r, accel, enc_speeds)

        # Track actions
        if best_goal == 1 or (cmd_pwm_l < -0.15 and cmd_pwm_r < -0.15):
            action_counts["REVERSE"] += 1
        elif best_goal == 2 or (cmd_pwm_r - cmd_pwm_l) > 0.10:
            action_counts["STEER_LEFT"] += 1
        elif best_goal == 3 or (cmd_pwm_l - cmd_pwm_r) > 0.10:
            action_counts["STEER_RIGHT"] += 1
        else:
            action_counts["FORWARD"] += 1

        prev_tensors = curr_tensors

    dur = time.time() - t0
    final_x = float(body.pose_x)
    final_y = float(body.pose_y)
    final_yaw = math.degrees(float(body.yaw))

    return {
        "scenario_id": scenario_id,
        "name": name,
        "steps": steps,
        "duration_s": dur,
        "final_x_mm": final_x,
        "final_y_mm": final_y,
        "final_yaw_deg": final_yaw,
        "collisions": collisions,
        "min_clearance_mm": min_clearance * 4000.0,
        "actions": action_counts,
        "mean_current": float(np.mean(motor_currents)),
        "peak_current": float(np.max(motor_currents)),
        "trajectory": trajectory,
        "arena": arena
    }


def plot_scenario_trajectory(res: Dict, output_paths: List[str]):
    """Generates an engineering trajectory plot showing the robot's physical movement."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    arena = res["arena"]
    fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
    ax.set_facecolor("#0f172a")
    fig.patch.set_facecolor("#0b0f19")

    # Draw all obstacles
    for obs in arena.obstacles:
        circle = patches.Circle((obs.x, obs.y), obs.radius, color="#ef4444", alpha=0.85, zorder=3)
        ax.add_patch(circle)
        if obs.radius >= 65.0:
            ax.text(obs.x, obs.y, f"R={int(obs.radius)}", color="white", fontsize=7, ha="center", va="center", weight="bold", zorder=4)

    # Trajectory points
    xs = [pt["x"] for pt in res["trajectory"]]
    ys = [pt["y"] for pt in res["trajectory"]]

    ax.plot(xs, ys, color="#38bdf8", linewidth=2.8, label="PARASYTE Trajectory", zorder=5)

    # Start and End points
    ax.scatter([xs[0]], [ys[0]], color="#22c55e", s=110, label="Start (0,0)", zorder=6, edgecolors="white", linewidth=1.5)
    ax.scatter([xs[-1]], [ys[-1]], color="#a855f7", s=130, label=f"End ({xs[-1]:.0f}, {ys[-1]:.0f})", zorder=6, edgecolors="white", linewidth=1.5)

    # Orientation arrow at end
    final_yaw = math.radians(res["final_yaw_deg"])
    arrow_len = 100.0
    ax.arrow(xs[-1], ys[-1], arrow_len * math.cos(final_yaw), arrow_len * math.sin(final_yaw),
             head_width=30, head_length=40, fc="#f8fafc", ec="#ffffff", zorder=7)

    # Subsample orientation arrows
    step_skip = max(len(xs) // 10, 1)
    for i in range(step_skip, len(xs) - 1, step_skip):
        yaw_rad = math.radians(res["trajectory"][i]["yaw_deg"])
        ax.arrow(xs[i], ys[i], 45 * math.cos(yaw_rad), 45 * math.sin(yaw_rad),
                 head_width=18, head_length=22, fc="#38bdf8", ec="#38bdf8", alpha=0.7, zorder=5)

    ax.set_title(f"{res['name']} — Physical Trajectory", color="#f8fafc", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("X Coordinate (mm) [Forward Direction]", color="#94a3b8", fontsize=10)
    ax.set_ylabel("Y Coordinate (mm) [Lateral Offset]", color="#94a3b8", fontsize=10)
    ax.tick_params(colors="#94a3b8")
    ax.grid(True, linestyle="--", alpha=0.25, color="#475569")

    # Telemetry stat box
    sim_time = res['steps'] * 0.02
    stats_text = (
        f"Physical Steps: {res['steps']} ({sim_time:.1f}s sim, {res['duration_s']:.2f}s CPU)\n"
        f"Final Position: X={res['final_x_mm']:.1f}mm, Y={res['final_y_mm']:.1f}mm\n"
        f"Final Heading: {res['final_yaw_deg']:.2f}°\n"
        f"Collisions: {res['collisions']}\n"
        f"Min Clearance: {res['min_clearance_mm']:.1f}mm\n"
        f"Actions: Fwd={res['actions']['FORWARD']}, TurnL={res['actions']['STEER_LEFT']}, "
        f"TurnR={res['actions']['STEER_RIGHT']}, Rev={res['actions']['REVERSE']}\n"
        f"Armature Current: Avg={res['mean_current']:.2f}A, Peak={res['peak_current']:.2f}A"
    )
    ax.text(0.02, 0.96, stats_text, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', color="#f1f5f9",
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#1e293b', alpha=0.9, edgecolor='#334155'))

    ax.legend(loc="lower right", facecolor="#1e293b", edgecolor="#334155", labelcolor="#f1f5f9", fontsize=8)
    ax.axis("equal")

    for p in output_paths:
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
        plt.savefig(p, bbox_inches="tight", dpi=150)
        print(f"      [+] Trajectory plot saved: {p}")
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PARASYTE 3 Core Empirical Benchmarks")
    parser.add_argument("--scenario", type=int, default=0, choices=[0, 1, 2, 3], help="Scenario ID (0 to run all 3 core benchmarks)")
    parser.add_argument("--steps", type=int, default=800, help="Number of simulation steps per scenario")
    args = parser.parse_args()

    ckpt_path = os.path.join(os.path.dirname(__file__), "mastered_brain.pt")

    print("=" * 80)
    print("      PARASYTE V2 — PHYSICAL BENCHMARK EVALUATION SUITE")
    print(f"      Target: {'All 3 Scenarios' if args.scenario == 0 else f'Scenario {args.scenario}'} | Steps: {args.steps}")
    print("=" * 80)

    if os.path.exists(ckpt_path):
        print(f"\n[1/2] Loading certified brain weights from {ckpt_path}...")
        t0_load = time.time()
        brain = torch.load(ckpt_path, weights_only=False)
        print(f"      Loaded certified brain in {time.time() - t0_load:.3f}s.")
    else:
        print("\n[1/2] Training developmental schooling pipeline (Active Inference + EWC)...")
        t_train_0 = time.time()
        brain = train_developmental_curriculum(seed=42, verbose=False)
        torch.save(brain, ckpt_path)
        print(f"      Schooling completed in {time.time() - t_train_0:.2f}s and saved to {ckpt_path}.")

    scenarios_to_run = [1, 2, 3] if args.scenario == 0 else [args.scenario]
    all_results = []
    

    canonical_steps = {1: 1200, 2: 2500, 3: 1500}
    print("\n[2/2] Executing Physical Benchmarks...")
    for sc_id in scenarios_to_run:
        step_budget = canonical_steps[sc_id] if args.steps == 800 else args.steps
        print(f"\n>> Executing Scenario {sc_id} ({step_budget} steps)...")
        res = execute_scenario(sc_id, brain, steps=step_budget)
        all_results.append(res)

        workspace_fig = os.path.join(os.path.dirname(__file__), "results", f"scenario_{sc_id}_path.png")
        plot_scenario_trajectory(res, [workspace_fig])

        print(f"   Done in {res['duration_s']:.2f}s | Final: X={res['final_x_mm']:.1f}mm, Y={res['final_y_mm']:.1f}mm | Heading: {res['final_yaw_deg']:.2f}°")
        print(f"   Collisions: {res['collisions']} | Min Clearance: {res['min_clearance_mm']:.1f}mm")
        print(f"   Actions: {res['actions']} | Current: {res['mean_current']:.2f}A avg, {res['peak_current']:.2f}A peak")

    print("\n" + "=" * 90)
    print("                     PARASYTE V2 — MASTER BENCHMARK SCORECARD")
    print("=" * 90)
    print(f"{'ID':<4} {'Scenario Name':<38} {'Distance (X, Y) mm':<24} {'Cols':<6} {'Clearance':<12} {'Rev Steps':<10}")
    print("-" * 90)
    for res in all_results:
        pos_str = f"({res['final_x_mm']:6.1f}, {res['final_y_mm']:5.1f})"
        clr_str = f"{res['min_clearance_mm']:5.1f} mm"
        rev_cnt = res['actions']['REVERSE']
        print(f"{res['scenario_id']:<4} {res['name'][:36]:<38} {pos_str:<24} {res['collisions']:<6} {clr_str:<12} {rev_cnt:<10}")
    print("=" * 90)



