"""
MPC-controlled quadrotor tracking a ground vehicle in Crazyflow.

Closed-loop MPC control with a keyboard-driven ground vehicle that has
simple friction physics. The drone tracks the ground vehicle.

Modes:
    1. HOVER  — drone takes off and holds position (automatic on start)
    2. TRACK  — press T to start tracking; WASD moves the vehicle, drone follows
                with a fixed altitude offset above
    3. LAND   — press L to descend onto the vehicle position

Keyboard (vehicle control):
    W / S  :  accelerate vehicle forward / backward  (+x / -x)
    A / D  :  accelerate vehicle left / right        (-y / +y)
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
    python mpc_controller_ground.py
"""

import threading

import mujoco
import numpy as np
from pynput import keyboard

from crazyflow.control import Control
from crazyflow.sim import Physics, Sim
from crazyflow.sim.integration import Integrator

from mpc_landing import MPCController, MPCConfig
from mpc_landing.reference import static_reference, tracking_reference, landing_reference


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

# Hover
HOVER_ALTITUDE = 1.0    # meters

# Ground vehicle physics (bicycle model)
MAX_SPEED = 1.0           # m/s forward
THROTTLE_ACCEL = 5.0      # m/s^2 when W held
BRAKE_DECEL = 8.0         # m/s^2 when S held (stronger than friction)
FRICTION_DECEL = 3.0      # m/s^2 coast deceleration
STEER_RATE = 2.0          # rad/s yaw rate at full steering input
V_STOP = 0.01             # m/s, snap-to-zero threshold

# Ground vehicle rendering
BOX_HALF_EXTENTS = np.array([0.20, 0.14, 0.04])  # 40cm x 28cm x 8cm
BOX_COLOR = np.array([0.2, 0.6, 1.0, 0.9])       # light blue

# Collision
DRONE_HALF_HEIGHT = 0.03  # approximate drone body half-height
COLLISION_MARGIN = 0.02   # horizontal tolerance

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


def draw_box(sim, pos, half_extents, rgba, yaw=0.0):
    """Draw a flat box marker at the given position.

    Args:
        sim: Crazyflow Sim instance.
        pos: np.array of shape (3,) -- center position in CF coordinates.
        half_extents: np.array of shape (3,) -- half-sizes [hx, hy, hz].
        rgba: np.array of shape (4,) -- color.
        yaw: rotation about the vertical (CF z) axis in radians.
    """
    if sim.viewer is None:
        return
    c, s = np.cos(yaw), np.sin(yaw)
    mat = np.array([
        c, -s, 0.0,
        s,  c, 0.0,
        0.0, 0.0, 1.0,
    ])
    sim.viewer.viewer.add_marker(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=half_extents,
        pos=pos,
        mat=mat,
        rgba=rgba,
    )


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
# Ground vehicle with simple friction physics
# ---------------------------------------------------------------------------
class GroundVehicle:
    """Keyboard-driven ground vehicle with bicycle-model steering."""

    def __init__(self):
        self.pos = np.array([0.0, 0.0, 0.0])  # MPC convention: [px, py, pz]
        self.vel = np.array([0.0, 0.0, 0.0])
        self.heading = 0.0  # rad, 0 = +x direction
        self.speed = 0.0    # scalar forward speed (m/s)

    def update(self, keys, dt):
        # Throttle / brake
        if keys.is_pressed("w"):
            self.speed += THROTTLE_ACCEL * dt
        elif keys.is_pressed("s"):
            self.speed -= BRAKE_DECEL * dt
        else:
            # Coast: friction deceleration toward zero
            if self.speed > V_STOP:
                self.speed = max(0.0, self.speed - FRICTION_DECEL * dt)
            elif self.speed < -V_STOP:
                self.speed = min(0.0, self.speed + FRICTION_DECEL * dt)
            else:
                self.speed = 0.0

        # Clamp speed
        self.speed = np.clip(self.speed, -MAX_SPEED, MAX_SPEED)

        # Steering (only turns when moving)
        if abs(self.speed) > V_STOP:
            if keys.is_pressed("a"):
                self.heading += STEER_RATE * dt
            if keys.is_pressed("d"):
                self.heading -= STEER_RATE * dt

        # Velocity from heading + speed
        self.vel[0] = self.speed * np.cos(self.heading)
        self.vel[2] = self.speed * np.sin(self.heading)

        # Update position
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

    # Ground vehicle
    vehicle = GroundVehicle()

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

    print("\n=== MPC Quadrotor + Ground Vehicle (Crazyflow) ===")
    print(f"Drone will hover at {HOVER_ALTITUDE}m above origin.")
    print()
    print("Controls:")
    print("  W/S : accelerate vehicle forward/back")
    print("  A/D : accelerate vehicle left/right")
    print("  T   : switch to TRACK mode (drone follows vehicle)")
    print("  L   : switch to LAND mode (drone descends onto vehicle)")
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
                print(f"\n>>> Mode: TRACK -- use WASD to drive vehicle")
            elif req == "land" and phase != LAND:
                phase = LAND
                print(f"\n>>> Mode: LAND -- descending onto vehicle")
            elif req == "hover":
                phase = HOVER
                hover_target = [0.0, HOVER_ALTITUDE, 0.0]
                vehicle.pos[:] = 0.0
                vehicle.vel[:] = 0.0
                vehicle.speed = 0.0
                vehicle.heading = 0.0
                print(f"\n>>> Mode: HOVER -- returning to origin")

            # --- Read drone state ---
            pos = np.array(sim.data.states.pos[0, 0])
            vel = np.array(sim.data.states.vel[0, 0])
            x0 = cf_to_mpc_state(pos, vel)

            # --- Update ground vehicle ---
            vehicle_state = vehicle.update(keys, CONTROL_DT)

            # --- Build reference ---
            if phase == HOVER:
                ref = static_reference(hover_target, N, CONTROL_DT)
            elif phase == TRACK:
                drone_state = {
                    "pos": [x0[0], x0[2], x0[4]],
                    "vel": [x0[1], x0[3], x0[5]],
                }
                ref = tracking_reference(drone_state, vehicle_state, N, CONTROL_DT)
            elif phase == LAND:
                drone_state = {
                    "pos": [x0[0], x0[2], x0[4]],
                    "vel": [x0[1], x0[3], x0[5]],
                }
                ref = landing_reference(drone_state, vehicle_state, N, CONTROL_DT)

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

            # --- Collision: clamp drone onto vehicle platform ---
            drone_pos = np.array(sim.data.states.pos[0, 0])  # CF [x, y, z]
            veh_cf_x = vehicle.pos[0]   # MPC px = CF x
            veh_cf_y = vehicle.pos[2]   # MPC pz = CF y
            platform_top = 2 * BOX_HALF_EXTENTS[2] + DRONE_HALF_HEIGHT

            in_footprint = (
                abs(drone_pos[0] - veh_cf_x) < BOX_HALF_EXTENTS[0] + COLLISION_MARGIN
                and abs(drone_pos[1] - veh_cf_y) < BOX_HALF_EXTENTS[1] + COLLISION_MARGIN
            )
            if in_footprint and drone_pos[2] < platform_top:
                new_pos = sim.data.states.pos.at[0, 0, 2].set(platform_top)
                new_vel = sim.data.states.vel
                if float(sim.data.states.vel[0, 0, 2]) < 0:
                    new_vel = new_vel.at[0, 0, 2].set(0.0)
                sim.data = sim.data.replace(
                    states=sim.data.states.replace(pos=new_pos, vel=new_vel)
                )

            # --- Render ---
            step_count += 1
            if (step_count * FPS) % CONTROL_HZ < FPS:
                # Draw vehicle box (MPC [px,py,pz] -> CF [x,y,z] = [px,pz,py])
                vehicle_cf = np.array([
                    vehicle.pos[0],
                    vehicle.pos[2],
                    BOX_HALF_EXTENTS[2],  # sit on ground plane
                ])
                draw_box(sim, vehicle_cf, BOX_HALF_EXTENTS, BOX_COLOR, yaw=vehicle.heading)
                sim.render()

            # --- Log at 10Hz ---
            if step_count % 5 == 0:
                print(
                    f"\r[{phase:>5s}] "
                    f"drone=({pos[0]:+.2f},{pos[1]:+.2f},{pos[2]:+.2f})  "
                    f"vehicle=({vehicle.pos[0]:+.2f},{vehicle.pos[2]:+.2f},{vehicle.pos[1]:+.2f})  "
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
