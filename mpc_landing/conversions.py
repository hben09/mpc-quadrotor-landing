"""
Acceleration-to-PWM conversion for Betaflight angle mode.

Converts MPC world-frame acceleration commands [ax, ay, az] into
PWM values [roll, pitch, throttle, yaw] that Betaflight angle mode expects.

Used by both the CoppeliaSim sim supervisor and (eventually) the real
supervisor when MPC is integrated.

Coordinate convention (matches pid.py and reference.py):
    X = forward  -> pitch
    Y = up       -> throttle
    Z = lateral  -> roll
"""

import math


def clamp_pwm(pwm):
    """Clamp a PWM value to [1000, 2000]."""
    return max(1000.0, min(2000.0, pwm))


def accel_to_pwm(ax, ay, az, mass, g=9.81, max_angle=0.6109, max_thrust=0.49):
    """Convert MPC acceleration commands to Betaflight angle-mode PWM.

    The drone produces acceleration by tilting (roll/pitch) and thrusting.
    Given desired world-frame accelerations, we invert the thrust-vector
    geometry to find the required tilt angles and thrust magnitude.

    Args:
        ax: desired forward acceleration (m/s^2) — produces pitch
        ay: desired upward acceleration (m/s^2) — produces throttle
        az: desired lateral acceleration (m/s^2) — produces roll
        mass: drone mass (kg)
        g: gravitational acceleration (m/s^2)
        max_angle: Betaflight angle mode limit (rad), default 35 deg
        max_thrust: maximum thrust (N), default ~2:1 TWR for 25g drone

    Returns:
        (roll_pwm, pitch_pwm, throttle_pwm, yaw_pwm) each in [1000, 2000]
    """
    # Total thrust magnitude: compensate for gravity in Y (up) axis
    ay_total = ay + g
    T = mass * math.sqrt(ax**2 + ay_total**2 + az**2)

    # Desired tilt angles from thrust vector direction
    roll_des = math.atan2(-az, ay_total)
    pitch_des = math.atan2(ax, ay_total)

    # Map angles to PWM (1500 = center, +-500 = +-max_angle)
    roll_pwm = 1500.0 + (roll_des / max_angle) * 500.0
    pitch_pwm = 1500.0 + (pitch_des / max_angle) * 500.0

    # Map thrust to PWM (1000 = 0%, 2000 = 100% of max_thrust)
    throttle_pwm = 1000.0 + (T / max_thrust) * 1000.0

    # Yaw: hold heading (no yaw command from MPC)
    yaw_pwm = 1500.0

    return clamp_pwm(roll_pwm), clamp_pwm(pitch_pwm), clamp_pwm(throttle_pwm), clamp_pwm(yaw_pwm)
