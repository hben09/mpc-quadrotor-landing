"""Proportional yaw controller.

Converts a (current_yaw, target_yaw) pair into a yawrate (deg/s) command
compatible with cflib's send_setpoint(). Yaw dynamics are decoupled from
position in the world frame, so this controller is independent of the
position MPC.

All angles in radians unless suffixed with _deg.
"""

import numpy as np

DEFAULT_KP = 3.0  # 1/s — P gain on yaw error
DEFAULT_MAX_YAWRATE_DEG = 90.0  # matches hardware/teleop.py


def wrap_to_pi(angle):
    """Wrap angle (radians) to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def compute_yawrate(
    current_yaw,
    target_yaw,
    target_yaw_rate=0.0,
    kp=DEFAULT_KP,
    max_yawrate_deg=DEFAULT_MAX_YAWRATE_DEG,
):
    """Return yawrate command (deg/s, cflib convention) driving current → target.

    target_yaw_rate is a feedforward term (rad/s, OptiTrack CCW+) — pass the
    rate at which target_yaw is changing so the controller commands the
    setpoint's heading rate directly and the P term only corrects residual
    error.

    Sign: OptiTrack euler[1] is CCW+ from above; cflib yawrate (inferred from
    hardware/teleop.py q/e mapping) is CW+ from above. Output is negated so
    a positive error (need to rotate CCW) produces negative yawrate.
    """
    err = wrap_to_pi(target_yaw - current_yaw)
    yawrate_rad = kp * err + target_yaw_rate  # OptiTrack CCW+
    yawrate_deg = -np.degrees(yawrate_rad)  # → cflib CW+
    return float(np.clip(yawrate_deg, -max_yawrate_deg, max_yawrate_deg))
