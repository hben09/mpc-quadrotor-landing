"""
MPC-controlled quadrotor in CoppeliaSim.

Closed-loop MPC control with keyboard-driven target. The drone tracks a
"virtual Limo" that you drive with the keyboard — no physical Limo object
needed in the scene.

Modes:
    1. HOVER  — drone takes off and holds position (automatic on start)
    2. TRACK  — press T to start tracking; WASD moves the target, drone follows
                with a fixed altitude offset above
    3. LAND   — press L to descend onto the target position

Keyboard (target control):
    W / S  :  move target forward / backward  (sim +x / -x)
    A / D  :  move target left / right        (sim -y / +y)
    T      :  start tracking mode
    L      :  start landing mode
    H      :  return to hover at origin
    Esc    :  stop simulation

Coordinate mapping (CoppeliaSim z-up -> MPC convention):
    sim x  ->  MPC px  (horizontal 1)
    sim z  ->  MPC py  (altitude / vertical)
    sim y  ->  MPC pz  (horizontal 2)

Usage:
    cd sim/
    python mpc_controller.py
"""

import sys
import time
import math
import threading
from pathlib import Path

import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from pynput import keyboard

# Add src/ to path relative to this file so the script works from any cwd
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from mpc import MPCController, MPCConfig
from reference import static_reference, tracking_reference, landing_reference


# ---------------------------------------------------------------------------
# CoppeliaSim object name
# ---------------------------------------------------------------------------
DRONE_NAME = "/Quadcopter"

# ---------------------------------------------------------------------------
# Control parameters
# ---------------------------------------------------------------------------
CONTROL_HZ = 50
CONTROL_DT = 1.0 / CONTROL_HZ

# PWM constants (must match Lua flight controller)
PWM_CENTER = 1500
PWM_MIN = 1000
PWM_MAX = 2000
HOVER_PWM = 1502  # empirically determined from debug_axes.py

# Acceleration -> PWM gains (tuned empirically)
K_PITCH = 8.0     # PWM/(m/s²)
K_ROLL = 8.0
K_THRUST = 10.0   # PWM/(m/s²) — needs more authority for altitude tracking

# Virtual Limo parameters
TARGET_SPEED = 1.0         # m/s when key held
TRACKING_ALTITUDE = 0.5    # meters above target
HOVER_ALTITUDE = 1.0       # meters for initial hover


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def sim_to_mpc_state(pos, vel):
    """CoppeliaSim (x,y,z) z-up -> MPC [px,vx, py,vy, pz,vz]."""
    return np.array([
        pos[0], vel[0],   # px, vx  (sim x)
        pos[2], vel[2],   # py, vy  (sim z = altitude)
        pos[1], vel[1],   # pz, vz  (sim y)
    ])


def mpc_accel_to_pwm(ax, ay, az):
    """MPC accelerations -> PWM. Small-angle linearization around hover.

    Sign conventions (verified empirically with debug_axes.py):
        +pitch PWM → +sim x → +MPC px  ✓  (same sign)
        +roll  PWM → -sim y → -MPC pz  ✗  (inverted!)
        +thrust PWM → +sim z → +MPC py ✓  (same sign)
    """
    pitch_pwm = PWM_CENTER + K_PITCH * ax
    thrust_pwm = HOVER_PWM + K_THRUST * ay
    roll_pwm = PWM_CENTER - K_ROLL * az   # INVERTED: +az → -roll PWM
    yaw_pwm = PWM_CENTER

    # Limit tilt to ±50 PWM to prevent flipping
    MAX_TILT = 50
    return (
        clamp(roll_pwm, PWM_CENTER - MAX_TILT, PWM_CENTER + MAX_TILT),
        clamp(pitch_pwm, PWM_CENTER - MAX_TILT, PWM_CENTER + MAX_TILT),
        clamp(yaw_pwm, PWM_MIN, PWM_MAX),
        clamp(thrust_pwm, PWM_MIN, PWM_MAX),
    )


def send_commands(sim, roll, pitch, yaw, thrust):
    # Round to int — Lua debug print uses %d which requires integer values
    sim.setFloatSignal("cmd_roll", float(int(round(roll))))
    sim.setFloatSignal("cmd_pitch", float(int(round(pitch))))
    sim.setFloatSignal("cmd_yaw", float(int(round(yaw))))
    sim.setFloatSignal("cmd_thrust", float(int(round(thrust))))


# ---------------------------------------------------------------------------
# Keyboard input (reused from keyboard_teleop.py)
# ---------------------------------------------------------------------------
class KeyState:
    def __init__(self):
        self._pressed = set()
        self._lock = threading.Lock()
        self.exit_flag = False
        self.mode_request = None  # "track", "land", "hover"

    def on_press(self, key):
        try:
            k = key.char
        except AttributeError:
            k = key
        with self._lock:
            self._pressed.add(k)
        # Mode switches
        if hasattr(key, "char"):
            if key.char == "t":
                self.mode_request = "track"
            elif key.char == "l":
                self.mode_request = "land"
            elif key.char == "h":
                self.mode_request = "hover"
        if key == keyboard.Key.esc:
            self.exit_flag = True

    def on_release(self, key):
        try:
            k = key.char
        except AttributeError:
            k = key
        with self._lock:
            self._pressed.discard(k)

    def is_pressed(self, char):
        with self._lock:
            return char in self._pressed

    def pop_mode_request(self):
        req = self.mode_request
        self.mode_request = None
        return req


# ---------------------------------------------------------------------------
# Virtual Limo (keyboard-driven target on the ground)
# ---------------------------------------------------------------------------
class VirtualLimo:
    """A point target you drive with WASD. No physics — just position + velocity."""

    def __init__(self):
        # Position in MPC convention: [horizontal1, altitude(ground), horizontal2]
        self.pos = np.array([0.0, 0.0, 0.0])
        self.vel = np.array([0.0, 0.0, 0.0])

    def update(self, keys, dt):
        """Move based on keyboard. Returns state dict for reference.py."""
        vx, vz = 0.0, 0.0

        if keys.is_pressed("w"):
            vx += TARGET_SPEED
        if keys.is_pressed("s"):
            vx -= TARGET_SPEED
        if keys.is_pressed("a"):
            vz += TARGET_SPEED
        if keys.is_pressed("d"):
            vz -= TARGET_SPEED

        self.vel = np.array([vx, 0.0, vz])
        self.pos += self.vel * dt

        return {
            "pos": self.pos.tolist(),
            "vel": self.vel.tolist(),
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Connecting to CoppeliaSim...")
    client = RemoteAPIClient()
    sim = client.getObject("sim")

    drone_handle = sim.getObject(DRONE_NAME)
    print(f"Found drone: {DRONE_NAME}")

    # MPC
    config = MPCConfig(dt=CONTROL_DT, horizon=25)
    mpc = MPCController(config)
    N = config.horizon
    print(f"MPC: horizon={N}, dt={CONTROL_DT}s")

    # Virtual Limo + visible marker in the scene
    limo = VirtualLimo()
    target_marker = sim.createPrimitiveShape(2, [0.1, 0.1, 0.1])  # 2 = sphere
    sim.setObjectAlias(target_marker, "Target")
    sim.setShapeColor(target_marker, None, 0, [1.0, 0.2, 0.2])  # 0 = ambient_diffuse
    sim.setObjectInt32Param(target_marker, sim.shapeintparam_static, 1)
    sim.setObjectInt32Param(target_marker, sim.shapeintparam_respondable, 0)

    # Keyboard
    keys = KeyState()
    listener = keyboard.Listener(on_press=keys.on_press, on_release=keys.on_release)
    listener.start()

    # Phase
    HOVER = "hover"
    TRACK = "track"
    LAND = "land"
    phase = HOVER
    hover_target = [0.0, HOVER_ALTITUDE, 0.0]

    # Start sim — free-running mode (matching keyboard_teleop.py)
    sim.startSimulation()
    time.sleep(0.2)

    print("\n=== MPC Quadrotor Controller ===")
    print("Drone will hover at 1m above origin.")
    print()
    print("Controls:")
    print("  W/S : move target forward/back")
    print("  A/D : move target left/right")
    print("  T   : switch to TRACK mode (drone follows target)")
    print("  L   : switch to LAND mode (drone descends onto target)")
    print("  H   : return to HOVER at origin")
    print("  Esc : quit")
    print("=" * 35)

    step_count = 0
    try:
        while sim.getSimulationState() != sim.simulation_stopped and not keys.exit_flag:
            # --- Mode switches ---
            req = keys.pop_mode_request()
            if req == "track" and phase != TRACK:
                phase = TRACK
                print(f"\n>>> Mode: TRACK — use WASD to move target")
            elif req == "land" and phase != LAND:
                phase = LAND
                print(f"\n>>> Mode: LAND — descending onto target")
            elif req == "hover":
                phase = HOVER
                hover_target = [0.0, HOVER_ALTITUDE, 0.0]
                limo.pos[:] = 0.0
                print(f"\n>>> Mode: HOVER — returning to origin")

            # --- Read drone state ---
            pos = sim.getObjectPosition(drone_handle, sim.handle_world)
            vel, _ = sim.getObjectVelocity(drone_handle)
            x0 = sim_to_mpc_state(pos, vel)

            # --- Update virtual Limo ---
            limo_state = limo.update(keys, CONTROL_DT)

            # --- Build reference ---
            if phase == HOVER:
                ref = static_reference(hover_target, N, CONTROL_DT)

            elif phase == TRACK:
                drone_state = {
                    "pos": [x0[0], x0[2], x0[4]],
                    "vel": [x0[1], x0[3], x0[5]],
                }
                ref = tracking_reference(drone_state, limo_state, N, CONTROL_DT)

            elif phase == LAND:
                drone_state = {
                    "pos": [x0[0], x0[2], x0[4]],
                    "vel": [x0[1], x0[3], x0[5]],
                }
                ref = landing_reference(drone_state, limo_state, N, CONTROL_DT)

            # --- Move target marker (MPC convention -> sim: x=px, y=pz, z=py) ---
            sim.setObjectPosition(target_marker, sim.handle_world,
                                  [limo.pos[0], limo.pos[2], limo.pos[1]])

            # --- Solve MPC ---
            u_opt = mpc.compute(x0, ref)
            ax, ay, az = u_opt

            # --- Send PWM ---
            roll_pwm, pitch_pwm, yaw_pwm, thrust_pwm = mpc_accel_to_pwm(ax, ay, az)
            send_commands(sim, roll_pwm, pitch_pwm, yaw_pwm, thrust_pwm)

            # --- Log EVERY step for debugging ---
            step_count += 1
            if step_count % 5 == 0:  # 10 Hz logging
                print(
                    f"[{phase:>5s}] "
                    f"sim=({pos[0]:+.2f},{pos[1]:+.2f},{pos[2]:+.2f})  "
                    f"mpc_state=({x0[0]:+.2f},{x0[2]:+.2f},{x0[4]:+.2f})  "
                    f"accel=({ax:+.2f},{ay:+.2f},{az:+.2f})  "
                    f"PWM: R={roll_pwm:.0f} P={pitch_pwm:.0f} T={thrust_pwm:.0f}"
                )

            time.sleep(CONTROL_DT)

            # Ground platform
            sim.setFloatSignal('ground_v', 0.5)


    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        listener.stop()
        send_commands(sim, PWM_CENTER, PWM_CENTER, PWM_CENTER, PWM_MIN)
        time.sleep(0.1)
        sim.removeObject(target_marker)
        sim.stopSimulation()
        print("\nSimulation stopped.")


if __name__ == "__main__":
    main()
