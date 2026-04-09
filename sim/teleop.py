"""Keyboard teleoperation for a Crazyflie in crazyflow.

Uses attitude control to match cflib's send_setpoint(roll, pitch, yawrate, thrust).

Controls:
    W/S     - pitch forward / backward
    A/D     - roll left / right
    Shift   - increase thrust
    Z       - decrease thrust
    Q/E     - yaw left / right
    R       - reset (level + hover thrust)
    ESC     - quit
"""

import numpy as np
from pynput import keyboard

from crazyflow.control import Control
from crazyflow.sim import Physics, Sim
from crazyflow.sim.integration import Integrator

MAX_ROLL = 0.5  # rad (~28 deg)
MAX_PITCH = 0.5  # rad (~28 deg)
RAMP_SPEED = 3.0  # rad/s — how fast roll/pitch ramps up/down
YAW_SPEED = 1.0  # rad/s
THRUST_STEP = 0.05  # N per second


def main():
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
    hover_thrust = mass * 9.81

    pressed: set[str] = set()
    running = True

    special_keys = {
        keyboard.Key.space: "_space",
        keyboard.Key.shift: "_shift",
        keyboard.Key.shift_l: "_shift",
        keyboard.Key.shift_r: "_shift",
    }

    def on_press(key):
        if key in special_keys:
            pressed.add(special_keys[key])
        else:
            try:
                pressed.add(key.char)
            except AttributeError:
                pass

    def on_release(key):
        if key == keyboard.Key.esc:
            nonlocal running
            running = False
            return False
        if key in special_keys:
            pressed.discard(special_keys[key])
        else:
            try:
                pressed.discard(key.char)
            except AttributeError:
                pass

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    thrust = hover_thrust
    thrust_max = float(sim.data.controls.attitude.params.thrust_max)
    roll = 0.0
    pitch = 0.0
    yaw = 0.0
    dt = 1.0 / sim.control_freq
    fps = 60

    print(f"Teleop active — hover thrust: {hover_thrust:.3f} N")
    print("W/S pitch, A/D roll, Z/Shift thrust, Q/E yaw, R reset, ESC quit")

    step = 0
    while running:
        # Reset
        if "r" in pressed:
            thrust = hover_thrust
            yaw = 0.0

        # Roll/pitch ramp toward target while key held, ramp back to 0 on release
        roll_target = 0.0
        pitch_target = 0.0
        if "a" in pressed:
            roll_target = -MAX_ROLL
        if "d" in pressed:
            roll_target = MAX_ROLL
        if "w" in pressed:
            pitch_target = MAX_PITCH
        if "s" in pressed:
            pitch_target = -MAX_PITCH

        if roll < roll_target:
            roll = min(roll + RAMP_SPEED * dt, roll_target)
        elif roll > roll_target:
            roll = max(roll - RAMP_SPEED * dt, roll_target)

        if pitch < pitch_target:
            pitch = min(pitch + RAMP_SPEED * dt, pitch_target)
        elif pitch > pitch_target:
            pitch = max(pitch - RAMP_SPEED * dt, pitch_target)

        # Thrust (holds value like a throttle)
        if "_shift" in pressed:
            thrust += THRUST_STEP * dt
        if "z" in pressed:
            thrust -= THRUST_STEP * dt
        thrust = max(thrust, 0.0)

        # Yaw (Q/E act as yaw rate — accumulates into angle for crazyflow)
        yaw_rate = 0.0
        if "q" in pressed:
            yaw_rate = YAW_SPEED
        if "e" in pressed:
            yaw_rate = -YAW_SPEED
        yaw += yaw_rate * dt

        # Attitude command: [roll, pitch, yaw, collective_thrust]
        cmd = np.zeros((sim.n_worlds, sim.n_drones, 4))
        cmd[..., 0] = roll
        cmd[..., 1] = pitch
        cmd[..., 2] = yaw
        cmd[..., 3] = thrust

        sim.attitude_control(cmd)
        sim.step(sim.freq // sim.control_freq)

        pwm = int(np.clip(thrust / thrust_max * 65535, 0, 65535))
        if ((step * fps) % sim.control_freq) < fps:
            sim.render()
            print(f"\rroll={roll:+.2f}  pitch={pitch:+.2f}  yaw_rate={yaw_rate:+.1f}  thrust={thrust:.3f} ({pwm})", end="")

        step += 1

    sim.close()
    listener.stop()
    print("Done.")


if __name__ == "__main__":
    main()
