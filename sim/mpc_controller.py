"""
MPC-controlled quadrotor in Crazyflow.

Closed-loop MPC control with keyboard-driven target. The drone tracks a
"virtual Limo" that you drive with the keyboard.

Modes:
    1. HOVER  — drone takes off and holds position (automatic on start)
    2. TRACK  — press T to start tracking; WASD moves the target, drone follows
                with a fixed altitude offset above
    3. LAND   — press L to descend onto the target position

Keyboard (target control):
    W / S  :  move target forward / backward  (+x / -x)
    A / D  :  move target left / right        (-y / +y)
    T      :  start tracking mode
    L      :  start landing mode
    H      :  return to hover at origin
    Esc    :  stop simulation

Coordinate mapping (Crazyflow z-up -> MPC convention):
    CF x  ->  MPC px  (horizontal 1 / forward)
    CF z  ->  MPC py  (altitude / vertical)
    CF y  ->  MPC pz  (horizontal 2 / lateral)

Usage:
    cd sim/
    python mpc_controller.py
"""

import threading

import numpy as np
from pynput import keyboard

from crazyflow.control import Control
from crazyflow.sim import Physics, Sim
from crazyflow.sim.integration import Integrator
from crazyflow.sim.visualize import draw_points

from mpc_landing import MPCController, MPCConfig
from mpc_landing.guidance import static_reference, tracking_reference, landing_reference


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
G = 9.81

# Timing
SIM_FREQ = 500          # Crazyflow physics rate (Hz)
CONTROL_HZ = 50         # MPC solve rate (Hz)
CONTROL_DT = 1.0 / CONTROL_HZ
STEPS_PER_MPC = SIM_FREQ // CONTROL_HZ  # 10 sim steps per MPC solve

# Attitude limits
MAX_TILT = 0.5  # rad (~28 deg), matching teleop.py

# Virtual Limo parameters
TARGET_SPEED = 1.0      # m/s when key held
HOVER_ALTITUDE = 1.0    # meters

# Rendering
FPS = 30


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def cf_to_mpc_state(pos, vel):
    """Crazyflow (x, y, z) z-up -> MPC [px, vx, py, vy, pz, vz]."""
    return np.array([
        pos[0], vel[0],   # px, vx  (CF x = forward)
        pos[2], vel[2],   # py, vy  (CF z = altitude)
        pos[1], vel[1],   # pz, vz  (CF y = lateral)
    ])


def mpc_accel_to_attitude(ax, ay, az, mass, thrust_max):
    """MPC accelerations -> Crazyflow attitude command [roll, pitch, yaw, thrust].

    Small-angle linearization around hover:
        pitch =  ax / g   (positive ax -> pitch forward -> +x -> +px)
        roll  = -az / g   (positive az -> +pz -> +CF_y -> need negative roll)
        thrust = mass * (g + ay)
    """
    pitch = np.clip(ax / G, -MAX_TILT, MAX_TILT)
    roll = np.clip(-az / G, -MAX_TILT, MAX_TILT)
    yaw = 0.0
    thrust = np.clip(mass * (G + ay), 0.0, thrust_max)
    return np.array([roll, pitch, yaw, thrust])


# ---------------------------------------------------------------------------
# Keyboard input
# ---------------------------------------------------------------------------
class KeyState:
    def __init__(self):
        self._pressed = set()
        self._lock = threading.Lock()
        self.exit_flag = False
        self.mode_request = None

    def on_press(self, key):
        try:
            k = key.char
        except AttributeError:
            k = key
        with self._lock:
            self._pressed.add(k)
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
    """A point target you drive with WASD. No physics -- just position + velocity."""

    def __init__(self):
        self.pos = np.array([0.0, 0.0, 0.0])  # MPC convention: [px, py, pz]
        self.vel = np.array([0.0, 0.0, 0.0])

    def update(self, keys, dt):
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
    print("Initializing Crazyflow sim...")
    sim = Sim(
        n_worlds=1,
        n_drones=1,
        physics=Physics.so_rpy_rotor_drag,
        control=Control.attitude,
        integrator=Integrator.rk4,
        freq=SIM_FREQ,
        attitude_freq=SIM_FREQ,
        device="cpu",
    )
    sim.reset()

    mass = float(sim.data.params.mass[0, 0, 0])
    thrust_max = float(sim.data.controls.attitude.params.thrust_max) * 4  # per-motor -> collective

    # MPC
    config = MPCConfig(dt=CONTROL_DT, horizon=25)
    mpc = MPCController(config)
    N = config.horizon
    print(f"MPC: horizon={N}, dt={CONTROL_DT}s, mass={mass:.4f}kg")

    # Virtual Limo
    limo = VirtualLimo()

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

    print("\n=== MPC Quadrotor Controller (Crazyflow) ===")
    print(f"Drone will hover at {HOVER_ALTITUDE}m above origin.")
    print()
    print("Controls:")
    print("  W/S : move target forward/back")
    print("  A/D : move target left/right")
    print("  T   : switch to TRACK mode (drone follows target)")
    print("  L   : switch to LAND mode (drone descends onto target)")
    print("  H   : return to HOVER at origin")
    print("  Esc : quit")
    print("=" * 48)

    step_count = 0
    try:
        while not keys.exit_flag:
            # --- Mode switches ---
            req = keys.pop_mode_request()
            if req == "track" and phase != TRACK:
                phase = TRACK
                print(f"\n>>> Mode: TRACK -- use WASD to move target")
            elif req == "land" and phase != LAND:
                phase = LAND
                print(f"\n>>> Mode: LAND -- descending onto target")
            elif req == "hover":
                phase = HOVER
                hover_target = [0.0, HOVER_ALTITUDE, 0.0]
                limo.pos[:] = 0.0
                print(f"\n>>> Mode: HOVER -- returning to origin")

            # --- Read drone state ---
            pos = np.array(sim.data.states.pos[0, 0])
            vel = np.array(sim.data.states.vel[0, 0])
            x0 = cf_to_mpc_state(pos, vel)

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

            # --- Solve MPC ---
            u_opt = mpc.compute(x0, ref)
            ax, ay, az = u_opt

            # --- Convert to attitude command ---
            attitude = mpc_accel_to_attitude(ax, ay, az, mass, thrust_max)
            cmd = np.zeros((sim.n_worlds, sim.n_drones, 4))
            cmd[0, 0, :] = attitude

            # --- Step sim (10 physics steps = 1 MPC step at 50Hz) ---
            sim.attitude_control(cmd)
            sim.step(STEPS_PER_MPC)

            # --- Render ---
            step_count += 1
            if (step_count * FPS) % CONTROL_HZ < FPS:
                # Draw target marker (MPC [px,py,pz] -> CF [x,y,z] = [px,pz,py])
                target_cf = np.array([[limo.pos[0], limo.pos[2], limo.pos[1]]])
                draw_points(sim, target_cf, rgba=np.array([1.0, 0.2, 0.2, 1.0]), size=0.03)
                sim.render()

            # --- Log at 10Hz ---
            if step_count % 5 == 0:
                print(
                    f"\r[{phase:>5s}] "
                    f"drone=({pos[0]:+.2f},{pos[1]:+.2f},{pos[2]:+.2f})  "
                    f"target=({limo.pos[0]:+.2f},{limo.pos[2]:+.2f},{limo.pos[1]:+.2f})  "
                    f"accel=({ax:+.2f},{ay:+.2f},{az:+.2f})  "
                    f"att: r={attitude[0]:+.3f} p={attitude[1]:+.3f} T={attitude[3]:.3f}",
                    end="",
                )

    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        listener.stop()
        sim.close()
        print("\nDone.")


if __name__ == "__main__":
    main()
