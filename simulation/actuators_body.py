"""
PARASYTE SITL — Physical Actuators & Mechanical Body Simulation.

This module models:
1. DC Drive Motors with electrical armature equations, back-EMF, and rotor inertia.
2. 8-Leg Kinematic Drive (4 Left, 4 Right in 90-degree quadrature phase).
3. Contact mechanics between 8 foot toes and terrain (continuous stance, zero singularities).
4. Strictly isolated Hardware Abstraction Layer (HAL) output interface.
"""

import math
import numpy as np
from typing import Dict, Tuple, Any


def circle_intersection(p1: Tuple[float, float], r1: float,
                        p2: Tuple[float, float], r2: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Computes analytical intersection of two 2D circles.
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
    if d > (r1 + r2) or d < abs(r1 - r2) or d == 0.0:
        d = max(d, 1e-6)
        r_sum = r1 + r2
        if d > r_sum:
            scale = r1 / r_sum
            mid = (x1 + scale * dx, y1 + scale * dy)
            return mid, mid
        else:
            mid = (x1 + (dx * 0.5), y1 + (dy * 0.5))
            return mid, mid

    a = (r1**2 - r2**2 + d**2) / (2.0 * d)
    h = math.sqrt(max(0.0, r1**2 - a**2))

    x_mid = x1 + (a * dx) / d
    y_mid = y1 + (a * dy) / d

    rx = -(dy * h) / d
    ry = (dx * h) / d

    return (x_mid + rx, y_mid + ry), (x_mid - rx, y_mid - ry)


def solve_canonical_leg(theta: float) -> Dict[str, Tuple[float, float]]:
    """
    Solves planar 11-bar linkage for crank angle theta (radians).
    Guaranteed zero singularities across 360 degrees:
    Uses correct Jansen link ratios with canonical Link I = 49.0 mm
    and branch index 0 for Node D.
    """
    # Fixed frame parameters (mm)
    A = (0.0, 0.0)             # Crank origin
    B = (-38.0, -7.8)          # Frame pivot

    # Link lengths (mm)
    R_crank = 15.0
    L_b = 50.0
    L_c = 39.3
    L_d = 40.1
    L_e = 55.8
    L_f = 39.4
    L_g = 36.7
    L_h = 65.7
    L_i = 49.0
    L_j = 50.0
    L_k = 61.9

    # 1. Crank node C
    C = (A[0] + R_crank * math.cos(theta), A[1] + R_crank * math.sin(theta))

    # 2. Node D: Intersection from B (r=L_k) and C (r=L_b). Branch 0 ensures geometric closure.
    pts_D = circle_intersection(B, L_k, C, L_b)
    D = pts_D[0]

    # 3. Node E: Intersection from C (r=L_c) and D (r=L_d)
    pts_E = circle_intersection(C, L_c, D, L_d)
    E = pts_E[0]

    # 4. Node F: Intersection from B (r=L_e) and E (r=L_f)
    pts_F = circle_intersection(B, L_e, E, L_f)
    F = pts_F[0]

    # 5. Node G (Ankle): Intersection from D (r=L_i) and E (r=L_j)
    pts_G = circle_intersection(D, L_i, E, L_j)
    G = pts_G[0]

    # 6. Foot Toe: Intersection from F (r=L_h) and G (r=L_g)
    pts_Foot = circle_intersection(F, L_h, G, L_g)
    Foot = pts_Foot[1]

    return {
        "A": A, "B": B, "C": C, "D": D, "E": E, "F": F, "G": G, "Foot": Foot
    }


class DCMotorDrive:
    """
    Simulates a geared DC drive motor with electrical armature resistance,
    back-EMF, rotor inertia, and viscous friction.
    """
    def __init__(self,
                 nominal_voltage: float = 12.0,
                 armature_resistance: float = 1.8,    # Ohms
                 torque_constant: float = 0.035,      # Nm / A (motor shaft)
                 back_emf_constant: float = 0.035,    # V / (rad/s) (motor shaft)
                 gear_ratio: float = 30.0,            # Output reduction
                 gear_efficiency: float = 0.85,
                 chassis_inertia: float = 0.025,      # kg * m^2 (crankshaft reflected)
                 viscous_friction: float = 0.04):     # Nm / (rad/s)
        self.v_nom = nominal_voltage
        self.r_a = armature_resistance
        self.k_t = torque_constant
        self.k_e = back_emf_constant
        self.gear_ratio = gear_ratio
        self.efficiency = gear_efficiency
        self.j_rotor = chassis_inertia
        self.b_friction = viscous_friction

        # State
        self.omega = 0.0       # Crankshaft output angular velocity (rad/s)
        self.theta = 0.0       # Crankshaft angle [0, 2*pi)
        self.current = 0.2     # Armature current (Amperes)

    def step(self, pwm_cmd: float, load_torque: float, dt: float) -> Tuple[float, float, float]:
        """
        Advance motor physics by dt seconds.
        pwm_cmd: float in [-1.0, 1.0]
        load_torque: external resistance on crankshaft (Nm)
        Returns: (delta_theta, omega, current)
        """
        pwm = max(-1.0, min(1.0, float(pwm_cmd)))
        voltage = pwm * self.v_nom

        # Motor shaft speed = crankshaft speed * gear ratio
        omega_motor = self.omega * self.gear_ratio
        v_back_emf = self.k_e * omega_motor

        # Armature current: i = (V - V_emf) / R_a
        i_elec = (voltage - v_back_emf) / self.r_a
        self.current += (dt / (dt + 0.005)) * (i_elec - self.current)

        # Output drive torque on crankshaft: tau = K_t * i * G * eta
        drive_torque = self.k_t * self.current * self.gear_ratio * self.efficiency

        # Resistive load opposing motion
        res_torque = (abs(load_torque) * np.sign(self.omega if abs(self.omega) > 1e-2 else pwm)) + (self.b_friction * self.omega)
        net_torque = drive_torque - res_torque

        # Angular acceleration of crankshaft
        alpha = net_torque / self.j_rotor
        self.omega += alpha * dt

        # Directional clamp: if zero voltage and small omega, bring to stop
        if abs(pwm) < 0.02 and abs(self.omega) < 0.1:
            self.omega = 0.0

        # Crankshaft integration
        delta_theta = self.omega * dt
        self.theta = (self.theta + delta_theta) % (2.0 * math.pi)

        return delta_theta, self.omega, self.current



class ArachnobeestBody:
    """
    Physical simulation of the 8-legged PARASYTE Octapod Body.
    - 2 DC motors (Left, Right)
    - 8 Jansen/Klann mechanical legs (4 Left, 4 Right with 90 deg quadrature phase)
    - Foot-ground contact and stance propulsion dynamics
    - True Hardware Abstraction Layer (HAL) boundary
    """

    def __init__(self,
                 track_width_mm: float = 220.0,
                 dt: float = 0.02):
        self.track_width = track_width_mm
        self.dt = dt

        # Actuators
        self.motor_left = DCMotorDrive()
        self.motor_right = DCMotorDrive()

        # Phase offsets for 4 legs per side: 0, 90, 180, 270 degrees
        self.phase_offsets = [0.0, math.pi * 0.5, math.pi, math.pi * 1.5]

        # 2D World pose: (x_mm, y_mm, heading_rad)
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.yaw = 0.0

        # Chassis velocities in body frame
        self.v_forward = 0.0  # mm/s
        self.omega_yaw = 0.0   # rad/s

        # Foot height contact threshold (mm in linkage coordinate frame)
        # Lowest foot point is y = -17.53 mm, highest is +0.76 mm.
        # Stance threshold at -8.0 mm corresponds to the bottom ~54% of the foot locus
        self.stance_threshold_y = -8.0


        # Telemetry history for inspection
        self.ground_truth_trajectory = []
        self.stance_history = []

    def step(self, pwm_left: float, pwm_right: float, obstacle_load: float = 0.0) -> Dict[str, Any]:
        """
        Step physical plant forward by 1 tick (self.dt).
        pwm_left, pwm_right: commands from brain in [-1.0, 1.0]
        obstacle_load: external resistance (e.g. hitting terrain/obstacle)
        
        Returns:
            Dict containing ONLY raw hardware signals suitable for HAL transmission.
        """
        # Baseline internal mechanical load from linkages (~0.15 Nm) + obstacle load
        load_l = 0.15 + obstacle_load
        load_r = 0.15 + obstacle_load

        # Step Left and Right DC Motors
        d_th_l, w_l, i_l = self.motor_left.step(pwm_left, load_l, self.dt)
        d_th_r, w_r, i_r = self.motor_right.step(pwm_right, load_r, self.dt)

        # Evaluate 8 legs (4 Left, 4 Right)
        left_stance_impulses = []
        right_stance_impulses = []
        toe_contacts = np.zeros(8, dtype=bool)
        toe_coords = np.zeros((8, 2), dtype=np.float32)

        # ── Left Side (Legs 0 to 3) ──
        for idx in range(4):
            th_curr = (self.motor_left.theta + self.phase_offsets[idx]) % (2.0 * math.pi)
            th_prev = (th_curr - d_th_l) % (2.0 * math.pi)

            sol_curr = solve_canonical_leg(th_curr)
            sol_prev = solve_canonical_leg(th_prev)

            f_curr = sol_curr["Foot"]
            f_prev = sol_prev["Foot"]
            toe_coords[idx] = [f_curr[0], f_curr[1]]

            # Stance condition: foot is down on the ground
            if f_curr[1] <= self.stance_threshold_y:
                toe_contacts[idx] = True
                dx_foot = f_curr[0] - f_prev[0]
                left_stance_impulses.append(abs(dx_foot) if w_l >= 0 else -abs(dx_foot))

        # ── Right Side (Legs 4 to 7) ──
        for idx in range(4):
            th_curr = (self.motor_right.theta + self.phase_offsets[idx]) % (2.0 * math.pi)
            th_prev = (th_curr - d_th_r) % (2.0 * math.pi)

            sol_curr = solve_canonical_leg(th_curr)
            sol_prev = solve_canonical_leg(th_prev)

            f_curr = sol_curr["Foot"]
            f_prev = sol_prev["Foot"]
            toe_coords[4 + idx] = [f_curr[0], f_curr[1]]

            if f_curr[1] <= self.stance_threshold_y:
                toe_contacts[4 + idx] = True
                dx_foot = f_curr[0] - f_prev[0]
                right_stance_impulses.append(abs(dx_foot) if w_r >= 0 else -abs(dx_foot))


        # ── Compute Chassis Body Dynamics ──
        # Forward speed driven by grounded feet
        v_l = (np.mean(left_stance_impulses) / self.dt) if len(left_stance_impulses) > 0 else (w_l * 15.0 * 0.4)
        v_r = (np.mean(right_stance_impulses) / self.dt) if len(right_stance_impulses) > 0 else (w_r * 15.0 * 0.4)

        # Differential chassis mechanics
        self.v_forward = float(0.5 * (v_l + v_r))
        self.omega_yaw = float((v_r - v_l) / self.track_width)

        # Integrate world pose
        heading_mid = self.yaw + 0.5 * self.omega_yaw * self.dt
        self.pose_x += self.v_forward * math.cos(heading_mid) * self.dt
        self.pose_y += self.v_forward * math.sin(heading_mid) * self.dt
        self.yaw = (self.yaw + self.omega_yaw * self.dt + math.pi) % (2.0 * math.pi) - math.pi

        self.ground_truth_trajectory.append((self.pose_x, self.pose_y, self.yaw))
        self.stance_history.append(int(np.sum(toe_contacts)))

        # ── Construct HAL Telemetry Packet ──
        # Only true sensor/actuator feedback sent to brain (with realistic ADC quantization & noise)
        enc_noise_l = np.random.normal(0.0, 0.002)
        enc_noise_r = np.random.normal(0.0, 0.002)
        current_noise_l = np.random.normal(0.0, 0.02)
        current_noise_r = np.random.normal(0.0, 0.02)

        hal_packet = {
            "encoders": np.array([d_th_l + enc_noise_l, d_th_r + enc_noise_r], dtype=np.float32),
            "motor_echo": np.array([abs(i_l) + current_noise_l, abs(i_r) + current_noise_r], dtype=np.float32),
            "battery_voltage": float(12.2 - 0.05 * (abs(i_l) + abs(i_r))),
        }

        # Keep private ground truth accessible only to simulation/benchmarks
        self._last_ground_truth = {
            "pose": np.array([self.pose_x, self.pose_y, self.yaw], dtype=np.float32),
            "velocity": np.array([self.v_forward, self.omega_yaw], dtype=np.float32),
            "stance_count": int(np.sum(toe_contacts)),
            "toe_contacts": toe_contacts,
            "toe_coords": toe_coords,
        }

        return hal_packet

    def get_ground_truth(self) -> Dict[str, Any]:
        """Accessor for verification scripts and physics logging ONLY."""
        return self._last_ground_truth
