"""
PARASYTE Hardware Safety Guardrails & Physical Interlocks
=========================================================
Implements real-time physical protection for both SITL simulation and
physical microcontroller execution (ESP32/MCU).

Safety Invariants:
1. Slew-rate limiting to prevent back-EMF spikes and mechanical shock on Klann linkages.
2. Motor stall & overcurrent watchdog to prevent motor driver burnout.
3. Inversion/tilt cutoff to disarm drive when chassis tips beyond recoverable angles.
"""

import math
from typing import Tuple, Dict, Any


class SlewRateLimiter:
    """Limits the rate of change of motor PWM commands to protect gears and H-bridges."""
    def __init__(self, max_slew_per_sec: float = 3.0, dt: float = 0.02):
        self.max_delta = max_slew_per_sec * dt  # e.g., 3.0 * 0.02 = 0.06 per tick
        self.last_pwm_l = 0.0
        self.last_pwm_r = 0.0

    def reset(self):
        self.last_pwm_l = 0.0
        self.last_pwm_r = 0.0

    def limit(self, target_l: float, target_r: float) -> Tuple[float, float]:
        delta_l = target_l - self.last_pwm_l
        delta_r = target_r - self.last_pwm_r

        safe_delta_l = max(-self.max_delta, min(self.max_delta, delta_l))
        safe_delta_r = max(-self.max_delta, min(self.max_delta, delta_r))

        self.last_pwm_l += safe_delta_l
        self.last_pwm_r += safe_delta_r

        # Hard clamp to [-1.0, 1.0]
        self.last_pwm_l = max(-1.0, min(1.0, self.last_pwm_l))
        self.last_pwm_r = max(-1.0, min(1.0, self.last_pwm_r))

        return self.last_pwm_l, self.last_pwm_r


class StallWatchdog:
    """Monitors motor effort vs. encoder motion to detect and mitigate physical stalls."""
    def __init__(self, effort_thresh: float = 0.35, min_ticks_thresh: float = 0.005, max_stall_ticks: int = 15):
        self.effort_thresh = effort_thresh
        self.min_ticks_thresh = min_ticks_thresh
        self.max_stall_ticks = max_stall_ticks
        self.stall_counter_l = 0
        self.stall_counter_r = 0
        self.is_stalled = False

    def reset(self):
        self.stall_counter_l = 0
        self.stall_counter_r = 0
        self.is_stalled = False

    def check(self, pwm_l: float, pwm_r: float, enc_speed_l: float, enc_speed_r: float) -> bool:
        # Check left motor stall
        if abs(pwm_l) > self.effort_thresh and abs(enc_speed_l) < self.min_ticks_thresh:
            self.stall_counter_l += 1
        else:
            self.stall_counter_l = max(0, self.stall_counter_l - 1)

        # Check right motor stall
        if abs(pwm_r) > self.effort_thresh and abs(enc_speed_r) < self.min_ticks_thresh:
            self.stall_counter_r += 1
        else:
            self.stall_counter_r = max(0, self.stall_counter_r - 1)

        self.is_stalled = (self.stall_counter_l >= self.max_stall_ticks) or (self.stall_counter_r >= self.max_stall_ticks)
        return self.is_stalled


class TiltCutoff:
    """Shuts down motors if the chassis exceeds safe pitch/roll thresholds (tip-over prevention)."""
    def __init__(self, max_tilt_deg: float = 35.0):
        self.max_tilt_rad = math.radians(max_tilt_deg)
        self.is_tripped = False

    def reset(self):
        self.is_tripped = False

    def check(self, imu_accel: Tuple[float, float, float]) -> bool:
        # Compute pitch and roll from gravity vector
        ax, ay, az = imu_accel[0], imu_accel[1], imu_accel[2]
        pitch = math.atan2(ax, math.sqrt(ay**2 + az**2))
        roll = math.atan2(ay, math.sqrt(ax**2 + az**2))

        if abs(pitch) > self.max_tilt_rad or abs(roll) > self.max_tilt_rad:
            self.is_tripped = True
        else:
            self.is_tripped = False
        return self.is_tripped


class HardwareSafetyGuard:
    """
    Unified Hardware Safety Officer.
    Enforces slew-rate limits, stall mitigation, and tilt cutoffs before
    commands reach physical motor registers or simulation actuators.
    """
    def __init__(self, max_slew: float = 3.0, dt: float = 0.02, max_tilt_deg: float = 35.0):
        self.slew = SlewRateLimiter(max_slew_per_sec=max_slew, dt=dt)
        self.stall = StallWatchdog()
        self.tilt = TiltCutoff(max_tilt_deg=max_tilt_deg)
        self.safety_interlock = False
        self.interlock_reasons = []

    def reset(self):
        self.slew.reset()
        self.stall.reset()
        self.tilt.reset()
        self.safety_interlock = False
        self.interlock_reasons = []

    def filter_action(self, target_l: float, target_r: float,
                      imu_accel: Tuple[float, float, float] = (0.0, 0.0, 9.81),
                      enc_speeds: Tuple[float, float] = (0.0, 0.0)) -> Tuple[float, float, Dict[str, Any]]:
        self.interlock_reasons = []

        # 1. Check Tilt
        if self.tilt.check(imu_accel):
            self.safety_interlock = True
            self.interlock_reasons.append("TILT_EXCEEDED")
            return 0.0, 0.0, {"interlock": True, "reasons": self.interlock_reasons}

        # 2. Check Stall
        if self.stall.check(target_l, target_r, enc_speeds[0], enc_speeds[1]):
            self.interlock_reasons.append("MOTOR_STALL_DETECTED")
            # Cut power slightly to prevent thermal runaway while preserving disengagement
            target_l *= 0.5
            target_r *= 0.5

        # 3. Apply Slew-Rate Limiting
        safe_l, safe_r = self.slew.limit(target_l, target_r)

        return safe_l, safe_r, {
            "interlock": len(self.interlock_reasons) > 0,
            "reasons": self.interlock_reasons,
            "raw_l": target_l,
            "raw_r": target_r
        }
