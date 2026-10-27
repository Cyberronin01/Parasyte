"""
PARASYTE V2 — Full-Component SITL Testbed Configuration
Exact dimensions, Jansen linkage parameters, sensor specifications, and neural hyperparameters.
"""

import math

# ═══════════════════════════════════════════════════════════
# 1. MECHANICAL & KINEMATICS SPECIFICATIONS
# ═══════════════════════════════════════════════════════════
# Jansen 11-Bar Linkage Canonical "Holy Numbers" (in mm)
CRANK_RADIUS = 15.0       # Crank link 'a' rotating around origin (0, 0)
FIXED_PIVOT_X = -38.0     # Fixed frame pivot 'B' X offset
FIXED_PIVOT_Y = -7.8      # Fixed frame pivot 'B' Y offset

# Link lengths for circle-circle kinematic closure
LINK_B = 50.0             # Connecting link from crank C to node D
LINK_C = 39.3             # Upper triangle link C to node E
LINK_D = 40.1             # Triangle link D to node E
LINK_E = 55.8             # Rocker link fixed pivot B to node F
LINK_F = 39.4             # Intermediate link E to node F
LINK_G = 36.7             # Ankle to Foot Tip
LINK_H = 65.7             # Node F to Foot Tip
LINK_J = 50.0             # Node E to Node G (ankle)
LINK_K = 61.9             # Fixed pivot B to node D

# Octapod Chassis Geometry
NUM_LEGS_PER_SIDE = 4
TOTAL_LEGS = 8
CHASSIS_TRACK_WIDTH = 180.0   # Distance between Left and Right leg banks (mm)
CHASSIS_LENGTH = 240.0        # Length of chassis (mm)
ROBOT_MASS_KG = 1.25          # Total physical robot mass

# Crankshaft Phase Offsets (Quadrature 90-degree wave gait)
PHASE_OFFSETS = [0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0]

# Physical Limits & Motor Dynamics
MAX_MOTOR_RPM = 120.0         # Geared DC motor max crank RPM
MAX_ANGULAR_VEL = (MAX_MOTOR_RPM * 2.0 * math.pi) / 60.0  # ~12.56 rad/s
MOTOR_TORQUE_STALL = 2.5      # Stall torque in Nm
MOTOR_INERTIA = 0.005

# ═══════════════════════════════════════════════════════════
# 2. SENSOR EMULATION & DIMENSIONS (EXACT HARDWARE SPEC)
# ═══════════════════════════════════════════════════════════
# Time-of-Flight Sensor: VL53L5CX (8x8 = 64 depth zones)
TOF_GRID_SIZE = 8
TOF_NUM_ZONES = 64
TOF_FOV_HORIZONTAL_DEG = 45.0
TOF_FOV_VERTICAL_DEG = 45.0
TOF_MAX_RANGE_MM = 4000.0     # 4 meters max depth
TOF_NOISE_STD_MM = 15.0       # Realistic sensor measurement jitter

# IMU: MPU6050 (6-DOF)
IMU_CHANNELS = 6              # [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
IMU_ACCEL_NOISE_STD = 0.05    # m/s^2 noise
IMU_GYRO_NOISE_STD = 0.02     # rad/s noise
GRAVITY = 9.81                # m/s^2

# Proprioception: Dual Encoders
ENCODER_CHANNELS = 2          # [left_speed, right_speed] (rad/s normalized)
ENCODER_CPR = 360             # Counts per rotation
ENCODER_NOISE_STD = 0.02

# Motor Command Echo
MOTOR_CMD_CHANNELS = 2        # [pwm_left, pwm_right] in [-1.0, 1.0]

# ═══════════════════════════════════════════════════════════
# 3. NEURAL ARCHITECTURE DIMENSIONS (ARCHITECTURE V2 SPEC)
# ═══════════════════════════════════════════════════════════
# Layer 2: Perception Embeddings
DEPTH_EMB_DIM = 32
BALANCE_EMB_DIM = 16
PROPRIO_EMB_DIM = 16
MOTOR_EMB_DIM = 16
PERCEPTION_TOTAL_DIM = DEPTH_EMB_DIM + BALANCE_EMB_DIM + PROPRIO_EMB_DIM + MOTOR_EMB_DIM  # 80

# Layer 3: Cognition & Fusion
FUSION_STATE_DIM = 32         # Unified state representation
ACTION_DIM = 2                # [pwm_left, pwm_right]

# ═══════════════════════════════════════════════════════════
# 4. LEARNING RATES & ONLINE ADAPTATION
# ═══════════════════════════════════════════════════════════
LR_PERCEPTION = 0.001
LR_FUSION = 0.0003
LR_ACTION = 0.0001
LR_ACTION_STAGE2_LOCOMOTION = 0.00001  # 10x slower for locomotion during Stage 2 (EWC needs time)
GRAD_CLIP_NORM = 1.0

# ═══════════════════════════════════════════════════════════
# 5. ELASTIC WEIGHT CONSOLIDATION (EWC)
# ═══════════════════════════════════════════════════════════
EWC_LAMBDA = 15.0             # Gentle penalty: guides without thrashing (was 1000 — too aggressive)
EWC_FISHER_SAMPLES = 200

# ═══════════════════════════════════════════════════════════
# 6. STATISTICAL SURPRISE DETECTOR (FAILURE DISCOVERY)
# ═══════════════════════════════════════════════════════════
SURPRISE_THRESHOLD = 3.0      # 3-sigma anomaly threshold
SURPRISE_EMA_ALPHA = 0.01     # Exponential moving average rate
SURPRISE_WARMUP_STEPS = 300   # Steps before failure detection is active (more warmup now)
DEATH_COOLDOWN_STEPS = 60     # Cooldown frames after a death event

# ═══════════════════════════════════════════════════════════
# 7. DEVELOPMENTAL CURRICULUM TIMING
# ═══════════════════════════════════════════════════════════
STAGE1A_BABBLE_STEPS = 500        # Motor babbling (was 200 — more embodiment time)
MIN_WALK_STEPS = 1500             # Minimum walk training before convergence is even checked
WALK_CONVERGENCE_WINDOW = 300     # Must be stable for this many consecutive steps
WALK_CONVERGENCE_RATIO_TOL = 0.12 # fast_ema / slow_ema within 12% = plateau detected
STAGE2_STEER_STEPS = 1200        # Steering training steps after walk gate passes
STAGE3_COLLISION_STEPS = 500      # Collision anomaly test steps
FISHER_SAMPLE_WINDOW = 300        # Gradient samples for Fisher matrix (was 200)

# ═══════════════════════════════════════════════════════════
# 7. SIMULATION LOOP TIMINGS
# ═══════════════════════════════════════════════════════════
LOOP_FREQ_HZ = 50             # 50 Hz control loop (20 ms dt)
DT = 1.0 / LOOP_FREQ_HZ
PHYSICS_SUBSTEPS = 5          # Internal physics substeps per control tick
