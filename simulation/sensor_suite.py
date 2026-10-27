"""
PARASYTE SITL — Physical Sensor Suite & Hardware Abstraction Layer (HAL).

Simulates real onboard sensors:
1. VL53L5CX: 64-zone (8x8) direct Time-of-Flight matrix with optical raycasting and photon noise.
2. MPU-6050: 6-DOF IMU (3-axis accelerometer with gravity & foot contact shock + 3-axis gyro with chassis yaw & gait roll/pitch wobble).
3. HALBridge: Strictly separated hardware interface delivering standard sensor packets to the AI brain.
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Any


class SimulatedObstacle:
    """Circular cylinder or pillar obstacle in the arena with finite 3D height."""
    def __init__(self, x_mm: float, y_mm: float, radius_mm: float, name: str = "Pillar", height_mm: float = 350.0):
        self.x = x_mm
        self.y = y_mm
        self.radius = radius_mm
        self.name = name
        self.height = height_mm


class ArenaWorld:
    """Manages physical objects and boundary walls in the arena."""
    def __init__(self):
        self.obstacles: List[SimulatedObstacle] = [
            SimulatedObstacle(800.0, 0.0, 150.0, "Pillar 1", 350.0),
            SimulatedObstacle(1600.0, 250.0, 180.0, "Pillar 2", 350.0),
            SimulatedObstacle(2400.0, -200.0, 160.0, "Pillar 3", 350.0),
            SimulatedObstacle(3200.0, 50.0, 200.0, "Wall Barrier", 400.0),
        ]
        # Arena boundary box: X in [-500, 4500], Y in [-1500, 1500]
        self.x_min = -500.0
        self.x_max = 4500.0
        self.y_min = -1500.0
        self.y_max = 1500.0


class VL53L5CXSensor:
    """
    STMicroelectronics VL53L5CX 8x8 Multi-Zone Time-of-Flight Sensor.
    - 64 independent ranging zones (8 columns x 8 rows)
    - 45 degree horizontal and vertical FOV
    - 20 mm to 4000 mm detection range
    - Physically authentic 3D optical raycasting:
      * Vertical beam angles determine ground plane (floor) intersections.
      * Rows 5-7 capture the floor depth gradient ahead of the robot.
      * Rows 0-4 capture approaching obstacles, walls, and open space.
      * Obstacle height bounds allow upper beams to clear low obstacles.
    """
    def __init__(self,
                 fov_deg: float = 60.0,
                 max_range_mm: float = 4000.0,
                 min_range_mm: float = 20.0,
                 sensor_offset_x_mm: float = 120.0,
                 sensor_mount_z_mm: float = 120.0,
                 sensor_pitch_deg: float = -4.0,
                 sensor_yaw_deg: float = 0.0):
        self.fov_rad = math.radians(fov_deg)
        self.max_range = max_range_mm
        self.min_range = min_range_mm
        self.offset_x = sensor_offset_x_mm
        self.mount_z = sensor_mount_z_mm
        self.pitch_rad = math.radians(sensor_pitch_deg)
        self.yaw_offset_rad = math.radians(sensor_yaw_deg)

        # Precompute the 8 horizontal beam angles relative to sensor optical normal (col 0: left, col 7: right)
        self.h_angles = np.linspace(self.fov_rad / 2.0, -self.fov_rad / 2.0, 8)
        # Precompute the 8 vertical beam angles (row 0: upper horizon, row 7: floor)
        self.v_angles = np.linspace(self.fov_rad / 2.0, -self.fov_rad / 2.0, 8)

    def read(self, chassis_x: float, chassis_y: float, chassis_yaw: float,
             arena: ArenaWorld) -> np.ndarray:
        """
        Performs 64 full 3D optical raycasts from the front nose of the robot into the arena.
        Returns:
            np.ndarray of shape (64,), values in [0.0, 1.0] (normalized mm / 4000.0).
        """
        # Sensor origin in 3D: (sensor_x, sensor_y, sensor_z)
        sensor_x = chassis_x + self.offset_x * math.cos(chassis_yaw)
        sensor_y = chassis_y + self.offset_x * math.sin(chassis_yaw)
        sensor_z = self.mount_z

        raw_depth_matrix = np.full((8, 8), self.max_range, dtype=np.float32)

        for col_idx, h_ang in enumerate(self.h_angles):
            ray_yaw = chassis_yaw + self.yaw_offset_rad + h_ang
            dir_x = math.cos(ray_yaw)
            dir_y = math.sin(ray_yaw)

            # 1. 2D Horizontal candidate intersections
            obs_candidates = []
            for obs in arena.obstacles:
                dx = obs.x - sensor_x
                dy = obs.y - sensor_y
                proj = dx * dir_x + dy * dir_y
                if proj > 0:
                    d_perp_sq = (dx**2 + dy**2) - proj**2
                    r_sq = obs.radius**2
                    if d_perp_sq < r_sq:
                        d_intersect = proj - math.sqrt(max(0.0, r_sq - d_perp_sq))
                        if self.min_range < d_intersect < self.max_range:
                            obs_candidates.append((d_intersect, obs.height))

            # Arena boundary walls (height = 1200 mm)
            wall_dist = self.max_range
            if dir_x > 1e-4:
                d_wall = (arena.x_max - sensor_x) / dir_x
                if self.min_range < d_wall < wall_dist: wall_dist = d_wall
            elif dir_x < -1e-4:
                d_wall = (arena.x_min - sensor_x) / dir_x
                if self.min_range < d_wall < wall_dist: wall_dist = d_wall

            if dir_y > 1e-4:
                d_wall = (arena.y_max - sensor_y) / dir_y
                if self.min_range < d_wall < wall_dist: wall_dist = d_wall
            elif dir_y < -1e-4:
                d_wall = (arena.y_min - sensor_y) / dir_y
                if self.min_range < d_wall < wall_dist: wall_dist = d_wall

            # 2. Iterate each vertical row (0 = top horizon, 7 = floor)
            for row_idx, v_ang in enumerate(self.v_angles):
                phi = self.pitch_rad + v_ang
                cos_phi = math.cos(phi)
                sin_phi = math.sin(phi)
                tan_phi = math.tan(phi)

                closest_3d = self.max_range

                # A. Floor intersection (Ground plane at Z = 0)
                if sin_phi < -1e-3:
                    d_floor_3d = -sensor_z / sin_phi
                    if self.min_range < d_floor_3d < closest_3d:
                        closest_3d = d_floor_3d

                # B. Obstacle checks in 3D
                for d_h, obs_height in obs_candidates:
                    z_at_obs = sensor_z + d_h * tan_phi
                    if 0.0 <= z_at_obs <= obs_height:
                        d_obs_3d = d_h / max(cos_phi, 1e-3)
                        if self.min_range < d_obs_3d < closest_3d:
                            closest_3d = d_obs_3d

                # C. Wall boundary check in 3D
                if wall_dist < self.max_range:
                    z_at_wall = sensor_z + wall_dist * tan_phi
                    if z_at_wall >= 0.0:
                        d_wall_3d = wall_dist / max(cos_phi, 1e-3)
                        if self.min_range < d_wall_3d < closest_3d:
                            closest_3d = d_wall_3d

                # Photon shot noise: ~3 mm baseline + 0.5% distance
                noise = np.random.normal(0.0, 3.0 + 0.005 * closest_3d)
                measured_dist = np.clip(closest_3d + noise, self.min_range, self.max_range)
                raw_depth_matrix[row_idx, col_idx] = measured_dist

        # Flatten 8x8 -> 64 vector and normalize to [0.0, 1.0]
        normalized_64 = raw_depth_matrix.flatten() / self.max_range
        return normalized_64.astype(np.float32)


class MPU6050Sensor:
    """
    MPU-6050 6-DOF Inertial Measurement Unit (IMU).
    - 3-axis Accelerometer: [ax, ay, az] in m/s^2 (includes +9.81 m/s^2 gravity along Z)
    - 3-axis Gyroscope: [gx, gy, gz] in rad/s (chassis angular rate + gait vibration)
    - Includes realistic bias drift and white noise.
    """
    def __init__(self,
                 accel_noise_std: float = 0.08,    # m/s^2
                 gyro_noise_std: float = 0.005):   # rad/s
        self.accel_noise = accel_noise_std
        self.gyro_noise = gyro_noise_std

        # Slow sensor bias drift (random walk)
        self.accel_bias = np.random.normal(0.0, 0.02, size=3)
        self.gyro_bias = np.random.normal(0.0, 0.001, size=3)

        self.prev_vx = 0.0

    def read(self, vx_mms: float, omega_yaw_rads: float,
             toe_contacts: np.ndarray, dt: float) -> np.ndarray:
        """
        Produces 6-DOF IMU reading [ax, ay, az, gx, gy, gz].
        vx_mms: chassis forward speed in mm/s
        omega_yaw_rads: chassis yaw rate in rad/s
        toe_contacts: 8 boolean contact states (generate foot impact vibrations)
        """
        vx_ms = vx_mms / 1000.0  # Convert to m/s
        accel_forward = (vx_ms - self.prev_vx) / max(dt, 1e-4)
        self.prev_vx = vx_ms

        # Leg impact vibration: each leg transitioning into stance creates a micro-jolt
        contact_count = np.sum(toe_contacts)
        vibration_phase = contact_count * 0.15

        # ── Accelerometer (m/s^2) ──
        # ax: forward acceleration + walking vibration
        ax = accel_forward + 0.12 * math.cos(vibration_phase * 6.0) + self.accel_bias[0] + np.random.normal(0.0, self.accel_noise)
        # ay: centripetal acceleration (v * omega) + lateral leg rocking
        ay = (vx_ms * omega_yaw_rads) + 0.08 * math.sin(vibration_phase * 6.0) + self.accel_bias[1] + np.random.normal(0.0, self.accel_noise)
        # az: Earth gravity (9.81 m/s^2) + vertical footstep bouncing
        az = 9.81 + 0.25 * math.cos(vibration_phase * 12.0) + self.accel_bias[2] + np.random.normal(0.0, self.accel_noise)

        # ── Gyroscope (rad/s) ──
        # gx: roll rocking rate as weight shifts between left and right legs
        gx = 0.04 * math.sin(vibration_phase * 6.0) + self.gyro_bias[0] + np.random.normal(0.0, self.gyro_noise)
        # gy: pitch rocking rate as legs step fore and aft
        gy = 0.03 * math.cos(vibration_phase * 6.0) + self.gyro_bias[1] + np.random.normal(0.0, self.gyro_noise)
        # gz: yaw rotational velocity around vertical axis
        gz = omega_yaw_rads + self.gyro_bias[2] + np.random.normal(0.0, self.gyro_noise)

        return np.array([ax, ay, az, gx, gy, gz], dtype=np.float32)


class HALBridge:
    """
    Strict Hardware Abstraction Layer (HAL).
    Glues the physical simulation (Body + Sensors) to the AI Brain.
    Ensures that the brain can NEVER access ground truth state variables.
    """
    def __init__(self, body, arena: ArenaWorld = None):
        self.body = body
        self.arena = arena if arena is not None else ArenaWorld()
        self.tof_front = VL53L5CXSensor(sensor_offset_x_mm=120.0, sensor_yaw_deg=0.0)
        self.tof_rear = VL53L5CXSensor(sensor_offset_x_mm=-120.0, sensor_yaw_deg=180.0)
        self.tof_sensor = self.tof_front  # Backward compatibility alias
        self.imu_sensor = MPU6050Sensor()

    def step(self, pwm_left: float, pwm_right: float, obstacle_load: float = 0.0) -> Dict[str, np.ndarray]:
        """
        Advances the physical world by 1 control tick:
        - Evaluates true physical collisions with arena obstacles (elastic contact + motor stall).
        - Returns the exact sensory tensors for the AI Brain.
        """
        chassis_radius = 160.0  # mm
        pose_x = self.body.pose_x
        pose_y = self.body.pose_y

        collision_active = False
        effective_load = obstacle_load
        self.last_hit_obstacle = None

        # 1. Physical collision evaluation against arena obstacles
        for obs in self.arena.obstacles:
            dist = math.hypot(obs.x - pose_x, obs.y - pose_y)
            if dist < (obs.radius + chassis_radius):
                collision_active = True
                self.last_hit_obstacle = obs
                # Physical stall load on motors
                effective_load = max(effective_load, 1.8)
                # Block forward chassis penetration (elastic boundary)
                overlap = (obs.radius + chassis_radius) - dist
                nx = (pose_x - obs.x) / max(dist, 1e-4)
                ny = (pose_y - obs.y) / max(dist, 1e-4)
                self.body.pose_x += nx * overlap * 0.8
                self.body.pose_y += ny * overlap * 0.8
                self.body.v_forward = min(0.0, self.body.v_forward)

        # Also check arena boundary walls
        if pose_x < (self.arena.x_min + chassis_radius) or pose_x > (self.arena.x_max - chassis_radius):
            collision_active = True
            effective_load = max(effective_load, 1.8)
            self.body.v_forward = 0.0

        if pose_y < (self.arena.y_min + chassis_radius) or pose_y > (self.arena.y_max - chassis_radius):
            collision_active = True
            effective_load = max(effective_load, 1.8)
            self.body.v_forward = 0.0

        self.last_collision_gt = collision_active

        # 2. Step the physical actuators and body with the true mechanical load
        body_packet = self.body.step(pwm_left, pwm_right, effective_load)
        gt = self.body.get_ground_truth()

        # 3. Sample VL53L5CX Time-of-Flight sensors (Dual: Front & Rear)
        pose_x, pose_y, yaw = gt["pose"]
        depth_front_64 = self.tof_front.read(pose_x, pose_y, yaw, self.arena)
        depth_rear_64 = self.tof_rear.read(pose_x, pose_y, yaw, self.arena)

        # 4. Sample MPU-6050 6-DOF IMU from body motion & foot contacts
        vx_mms, omega_yaw = gt["velocity"]
        imu_6dof = self.imu_sensor.read(vx_mms, omega_yaw, gt["toe_contacts"], self.body.dt)

        # 5. Assemble the pristine HAL sensory packet
        brain_inputs = {
            "depth": depth_front_64,         # Standard alias for front depth
            "depth_front": depth_front_64,   # Explicit front 64-zone depth
            "depth_rear": depth_rear_64,     # Explicit rear 64-zone depth
            "imu": imu_6dof,
            "encoders": body_packet["encoders"],
            "motor_echo": body_packet["motor_echo"],
            "battery_voltage": body_packet["battery_voltage"],
        }
        return brain_inputs

