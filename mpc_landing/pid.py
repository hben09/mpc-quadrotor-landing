"""
PID controller from RASTIC safeguard system.
This is the baseline controller — will be replaced with MPC.
"""

import math
import time
import csv


class PID:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output


class PIDController:
    def __init__(self, log_file=None):
        self.log_file = log_file
        if self.log_file:
            with open(self.log_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['x', 'y', 'z', 'roll', 'pitch', 'throttle', 'yaw'])

        self.last_time = time.time()
        self.y_prev = 0.0
        self.x_actual = 0.0
        self.y_actual = 0.0
        self.z_actual = 0.0
        self.yaw_actual = 0.0
        self.pitch_actual = 0.0
        self.roll_actual = 0.0

        # PID gains — from RASTIC tuning
        self.pid_yaw = PID(kp=50, ki=0.0, kd=1.5)
        self.pid_pitch = PID(kp=75, ki=1.5, kd=25.0)
        self.pid_roll = PID(kp=75, ki=1.5, kd=15.0)
        self.pid_alt = PID(kp=10, ki=2.0, kd=5.0)

    def update_state(self, x, y, z, roll, pitch, yaw):
        self.x_actual = x
        self.y_actual = y
        self.z_actual = z
        self.yaw_actual = yaw
        self.pitch_actual = pitch
        self.roll_actual = roll

    def reset(self):
        self.pid_yaw.reset()
        self.pid_pitch.reset()
        self.pid_roll.reset()
        self.pid_alt.reset()
        self.last_time = time.time()

    def compute(self, goal):
        """Compute roll, pitch, throttle, yaw PWM values given a [x, y, z] goal.

        Returns:
            (roll, pitch, throttle, yaw) as PWM values centered at 1500.
        """
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time

        g = 9.81
        k_pos = 5

        x_error = goal[0] - self.x_actual
        y_error = goal[1] - self.y_actual
        z_error = self.z_actual - goal[2]

        if abs(x_error) < 0.05:
            x_error = 0.0
        if abs(z_error) < 0.05:
            z_error = 0.0

        pitch_target = math.atan2(k_pos * x_error, g)
        roll_target = math.atan2(-k_pos * z_error, g)

        pitch_correction = self.pid_pitch.compute(pitch_target - self.pitch_actual, dt)
        roll_correction = self.pid_roll.compute(roll_target - self.roll_actual, dt)

        pitch = pitch_correction + 1500
        roll = roll_correction + 1500

        velocity_actual = (self.y_actual - self.y_prev) / dt if dt > 0 else 0.0
        velocity_goal = 5 * y_error
        throttle_correction = self.pid_alt.compute(velocity_goal - velocity_actual, dt)
        hover_throttle = 1400
        throttle = int(min(max(hover_throttle + throttle_correction, 1000), 2000))
        self.y_prev = self.y_actual

        yaw = 1500

        if self.log_file:
            with open(self.log_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.x_actual, self.y_actual, self.z_actual,
                    roll, pitch, hover_throttle + throttle_correction, yaw,
                ])

        return roll, pitch, throttle, yaw
