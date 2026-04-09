"""Keyboard teleoperation for a Crazyflie in crazyflow.

Uses attitude control to match cflib's send_setpoint(roll, pitch, yawrate, thrust).

Controls:
    W/S     - pitch forward / backward
    A/D     - roll left / right
    Q/E     - yaw left / right
    1       - DISARM (0 thrust)
    2       - ARM (15000 thrust)
    3       - HOVER (35000 thrust)
    Space   - increase thrust (HOVER mode)
    Shift   - decrease thrust (HOVER mode)
    ESC     - quit
"""

import sys
import time
from threading import Lock

import numpy as np
from pynput import keyboard

from crazyflow.control import Control
from crazyflow.sim import Physics, Sim
from crazyflow.sim.integration import Integrator

MAX_ROLL = 15.0       # degrees
MAX_PITCH = 15.0      # degrees
MAX_YAWRATE = 60.0    # degrees/s
MAX_THRUST = 45000

# Discrete thrust modes
MODE_DISARM = 'DISARM'
MODE_ARM = 'ARM'
MODE_HOVER = 'HOVER'
MODE_THRUST = {MODE_DISARM: 0, MODE_ARM: 15000, MODE_HOVER: 35000}

# Shared state
pressed_keys = set()
keys_lock = Lock()
running = True


def on_press(key):
    global running
    if key == keyboard.Key.esc:
        running = False
        return False  # stop listener
    with keys_lock:
        try:
            pressed_keys.add(key.char.lower())
        except AttributeError:
            pressed_keys.add(key)


def on_release(key):
    with keys_lock:
        try:
            pressed_keys.discard(key.char.lower())
        except AttributeError:
            pressed_keys.discard(key)


def compute_setpoint(mode):
    with keys_lock:
        keys = set(pressed_keys)

    roll = 0.0
    pitch = 0.0
    yawrate = 0.0

    if 'w' in keys:
        pitch = MAX_PITCH
    if 's' in keys:
        pitch = -MAX_PITCH
    if 'a' in keys:
        roll = -MAX_ROLL
    if 'd' in keys:
        roll = MAX_ROLL
    if 'q' in keys:
        yawrate = -MAX_YAWRATE
    if 'e' in keys:
        yawrate = MAX_YAWRATE

    if '1' in keys:
        mode = MODE_DISARM
    elif '2' in keys:
        mode = MODE_ARM
    elif '3' in keys:
        mode = MODE_HOVER

    thrust = MODE_THRUST[mode]

    if mode == MODE_HOVER:
        if keyboard.Key.space in keys:
            thrust += 4000
        elif keyboard.Key.shift_l in keys or keyboard.Key.shift_r in keys or keyboard.Key.shift in keys:
            thrust -= 4000

    return roll, pitch, yawrate, thrust, mode


def pwm_to_thrust(pwm, mass):
    """Convert cflib PWM (0–65535) to collective thrust in Newtons."""
    hover_pwm = 35000
    hover_thrust = mass * 9.81
    return pwm / hover_pwm * hover_thrust


def main():
    global running

    sim = Sim(
        n_worlds=1,
        n_drones=1,
        physics=Physics.so_rpy_rotor_drag,
        control=Control.attitude,
        integrator=Integrator.rk4,
        freq=500,
        attitude_freq=500,
        device="cpu",
    )
    sim.reset()

    mass = float(sim.data.params.mass[0, 0, 0])

    print()
    print('=== Keyboard Control Active (Sim) ===')
    print('W/S = pitch | A/D = roll | Q/E = yaw')
    print('1 = DISARM (0) | 2 = ARM (15000) | 3 = HOVER (35000)')
    print('Space = thrust up | Shift = thrust down (HOVER mode)')
    print('Esc = quit')
    print()

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    dt = 0.02  # 50 Hz
    steps_per_iter = sim.freq // 50
    yaw_angle = 0.0
    mode = MODE_DISARM

    try:
        while running:
            roll, pitch, yawrate, thrust, mode = compute_setpoint(mode)

            # Convert cflib units to crazyflow units at the API boundary
            yaw_angle -= np.radians(yawrate) * dt

            cmd = np.zeros((sim.n_worlds, sim.n_drones, 4))
            cmd[..., 0] = np.radians(pitch)    # CF roll = forward/back (model rotated 90° from cflib)
            cmd[..., 1] = -np.radians(roll)   # CF pitch = left/right (negated: -cflib_roll = +CF_pitch)
            cmd[..., 2] = yaw_angle
            cmd[..., 3] = pwm_to_thrust(thrust, mass)

            sim.attitude_control(cmd)
            sim.step(steps_per_iter)
            sim.render()

            sys.stdout.write(
                f'\rMode: {mode:7s}  Roll: {roll:+6.1f}  Pitch: {pitch:+6.1f}  '
                f'Yaw: {yawrate:+6.1f}  Thrust: {thrust:5d}  '
            )
            sys.stdout.flush()
            time.sleep(dt)
    finally:
        sim.close()
        listener.stop()
        print('\nSim closed. Done.')


if __name__ == '__main__':
    main()
