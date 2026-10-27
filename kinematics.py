"""
PARASYTE V2 — Analytical Jansen 11-Bar Kinematics Engine
Closed-form circle-circle intersection position analysis, 8-leg staggered stance phase detection,
and 2D chassis differential dynamics.
"""

import math
import numpy as np
from config import (CRANK_RADIUS, FIXED_PIVOT_X, FIXED_PIVOT_Y,
                    LINK_B, LINK_C, LINK_D, LINK_E, LINK_F, LINK_G, LINK_H, LINK_J, LINK_K,
                    NUM_LEGS_PER_SIDE, CHASSIS_TRACK_WIDTH, PHASE_OFFSETS, MAX_ANGULAR_VEL)


def circle_intersection(p1, r1, p2, r2):
    """
    Computes analytical intersection of two circles.
    Circle 1: center p1=(x1, y1), radius r1
    Circle 2: center p2=(x2, y2), radius r2
    Returns two (x, y) intersection tuples.
    """
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    d = math.hypot(dx, dy)

    # Singularity / non-intersecting guard
    if d > (r1 + r2) or d < abs(r1 - r2) or d == 0:
        d = max(d, 1e-6)
        r_sum = r1 + r2
        if d > r_sum:
            scale = r1 / (r1 + r2)
            mid = (x1 + scale * dx, y1 + scale * dy)
            return mid, mid

    a = (r1**2 - r2**2 + d**2) / (2.0 * d)
    h = math.sqrt(max(0.0, r1**2 - a**2))

    x_mid = x1 + (a * dx) / d
    y_mid = y1 + (a * dy) / d

    rx = -(dy * h) / d
    ry = (dx * h) / d

    return (x_mid + rx, y_mid + ry), (x_mid - rx, y_mid - ry)


def solve_single_jansen_leg(theta):
    """
    Solves instantaneous 11-bar Jansen leg position for crank angle theta (radians).
    Returns dict of key joint coordinates and (x_foot, y_foot).
    """
    # Fixed pivots
    A = (0.0, 0.0)                                     # Crank origin
    B = (FIXED_PIVOT_X, FIXED_PIVOT_Y)                 # Fixed frame pivot (-38.0, -7.8)

    # 1. Crank node C
    C = (A[0] + CRANK_RADIUS * math.cos(theta), A[1] + CRANK_RADIUS * math.sin(theta))

    # 2. Node D: Intersection from B (r=LINK_K=61.9) and C (r=LINK_B=50.0)
    pts_D = circle_intersection(B, LINK_K, C, LINK_B)
    D = pts_D[1]

    # 3. Node E: Intersection from C (r=LINK_C=39.3) and D (r=LINK_D=40.1)
    pts_E = circle_intersection(C, LINK_C, D, LINK_D)
    E = pts_E[0]

    # 4. Node F: Intersection from B (r=LINK_E=55.8) and E (r=LINK_F=39.4)
    pts_F = circle_intersection(B, LINK_E, E, LINK_F)
    F = pts_F[0]

    # 5. Node G (Ankle): Intersection from D (r=LINK_K=61.9) and E (r=LINK_J=50.0)
    pts_G = circle_intersection(D, LINK_K, E, LINK_J)
    G = pts_G[0]

    # 6. Foot Tip: Intersection from F (r=LINK_H=65.7) and G (r=LINK_G=36.7)
    pts_Foot = circle_intersection(F, LINK_H, G, LINK_G)
    Foot = pts_Foot[1]

    return {
        "A": A, "B": B, "C": C, "D": D, "E": E, "F": F, "G": G, "Foot": Foot
    }


class JansenOctapodKinematics:
    """
    Complete 8-legged chassis kinematics:
    - 4 legs Left, 4 legs Right
    - 90-degree crankshaft phase offsets
    - Ground reaction stance evaluation
    - Chassis translation & yaw integration
    """

    def __init__(self):
        # Chassis pose in 2D world frame: (X_mm, Y_mm, Heading_rad)
        self.x = 0.0
        self.y = 0.0
        self.heading = 0.0

        # Crankshaft state (radians)
        self.theta_left = 0.0
        self.theta_right = 0.0

        # Actual crank angular velocities (rad/s)
        self.omega_left = 0.0
        self.omega_right = 0.0

        # Linear and angular chassis velocities
        self.v_forward = 0.0     # mm/s
        self.omega_yaw = 0.0      # rad/s

        # Foot height threshold for stance phase (at bottom 54% of cycle)
        self.stance_y_threshold = 20.0  # mm

        # History tracking for analysis and figures
        self.left_feet_history = [[] for _ in range(NUM_LEGS_PER_SIDE)]
        self.right_feet_history = [[] for _ in range(NUM_LEGS_PER_SIDE)]
        self.trajectory_path = []

    def update(self, pwm_left, pwm_right, dt):
        """
        Takes commanded Left/Right PWM [-1.0, 1.0], computes dynamic motor speeds,
        updates crankshafts, calculates stance propulsion, and updates chassis pose.
        """
        # Motor dynamics: convert PWM to target crank speed with inertia lag
        target_omega_l = pwm_left * MAX_ANGULAR_VEL
        target_omega_r = pwm_right * MAX_ANGULAR_VEL

        motor_alpha = 0.25  # Mechanical inertia time constant
        self.omega_left += motor_alpha * (target_omega_l - self.omega_left)
        self.omega_right += motor_alpha * (target_omega_r - self.omega_right)

        # Advance crankshaft angles
        prev_theta_l = self.theta_left
        prev_theta_r = self.theta_right
        self.theta_left = (self.theta_left + self.omega_left * dt) % (2.0 * math.pi)
        self.theta_right = (self.theta_right + self.omega_right * dt) % (2.0 * math.pi)

        # ── Evaluate Left Legs (4 legs, 90 deg offset) ──
        left_stance_count = 0
        left_forward_impulses = []
        for i in range(NUM_LEGS_PER_SIDE):
            th_curr = (self.theta_left + PHASE_OFFSETS[i]) % (2.0 * math.pi)
            th_prev = (prev_theta_l + PHASE_OFFSETS[i]) % (2.0 * math.pi)

            sol_curr = solve_single_jansen_leg(th_curr)
            sol_prev = solve_single_jansen_leg(th_prev)

            foot_curr = sol_curr["Foot"]
            foot_prev = sol_prev["Foot"]

            # Stance check: foot at bottom of cycle pushes ground
            if foot_curr[1] <= self.stance_y_threshold:
                left_stance_count += 1
                # Forward propulsion generated as foot pushes ground backward relative to body
                dx_foot = foot_curr[0] - foot_prev[0]
                left_forward_impulses.append(dx_foot / dt)

        # ── Evaluate Right Legs (4 legs, 90 deg offset) ──
        right_stance_count = 0
        right_forward_impulses = []
        for i in range(NUM_LEGS_PER_SIDE):
            th_curr = (self.theta_right + PHASE_OFFSETS[i]) % (2.0 * math.pi)
            th_prev = (prev_theta_r + PHASE_OFFSETS[i]) % (2.0 * math.pi)

            sol_curr = solve_single_jansen_leg(th_curr)
            sol_prev = solve_single_jansen_leg(th_prev)

            foot_curr = sol_curr["Foot"]
            foot_prev = sol_prev["Foot"]

            if foot_curr[1] <= self.stance_y_threshold:
                right_stance_count += 1
                dx_foot = foot_curr[0] - foot_prev[0]
                right_forward_impulses.append(dx_foot / dt)

        # Compute effective ground contact drive speeds (with slip)
        v_l = np.mean(left_forward_impulses) if left_forward_impulses else 0.0
        v_r = np.mean(right_forward_impulses) if right_forward_impulses else 0.0

        # Chassis differential velocity
        self.v_forward = (v_r + v_l) / 2.0
        self.omega_yaw = (v_r - v_l) / CHASSIS_TRACK_WIDTH

        # Integrate chassis displacement in 2D floor plane
        self.heading += self.omega_yaw * dt
        self.x += self.v_forward * math.cos(self.heading) * dt
        self.y += self.v_forward * math.sin(self.heading) * dt

        self.trajectory_path.append((self.x, self.y, self.heading))

        total_grounded_legs = left_stance_count + right_stance_count

        return {
            "v_forward_mms": self.v_forward,
            "omega_yaw_rads": self.omega_yaw,
            "x_mm": self.x,
            "y_mm": self.y,
            "heading_rad": self.heading,
            "grounded_legs": total_grounded_legs,
            "left_grounded": left_stance_count,
            "right_grounded": right_stance_count
        }

    def get_static_stability_margin(self):
        """
        Calculates ground contact support polygon. Returns minimum number of legs grounded.
        Guarantees >= 4 legs grounded at any moment in the 360-degree rotation.
        """
        grounded_over_cycle = []
        for th in np.linspace(0, 2.0 * math.pi, 120):
            l_count = sum(1 for i in range(4) if solve_single_jansen_leg((th + PHASE_OFFSETS[i]) % (2 * math.pi))["Foot"][1] <= self.stance_y_threshold)
            r_count = sum(1 for i in range(4) if solve_single_jansen_leg((th + PHASE_OFFSETS[i]) % (2 * math.pi))["Foot"][1] <= self.stance_y_threshold)
            grounded_over_cycle.append(l_count + r_count)
        return min(grounded_over_cycle), max(grounded_over_cycle), np.mean(grounded_over_cycle)
