"""
Crazyflow simulation for the landing teleop workflow.

This is a sim-side counterpart to the hardware landing controller, but with:
- a keyboard-driven manual tracking point controlled by WASD
- a separate fixed landing cardboard placed randomly on the ground near origin

Controls:
    Space  : start takeoff
    W / S  : move manual target forward / backward
    A / D  : move manual target left / right
    M      : track the manual target point
    T      : track the cardboard at fixed altitude
    L      : land on the cardboard
    O      : respawn cardboard near origin
    H      : hover at origin
    1-6    : select MPC parameter
    Up/Down: tune selected MPC parameter
    Esc    : quit

Usage:
    uv run sim-mpc-teleop-landing
"""

import threading

import mujoco
import numpy as np
from pynput import keyboard

from crazyflow.control import Control
from crazyflow.sim import Physics, Sim
from crazyflow.sim.integration import Integrator
from crazyflow.sim.visualize import draw_points

from mpc_landing import MPCController, MPCConfig
from mpc_landing.guidance import static_reference, landing_reference

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
G = 9.81

SIM_FREQ = 500
CONTROL_HZ = 50
CONTROL_DT = 1.0 / CONTROL_HZ
STEPS_PER_MPC = SIM_FREQ // CONTROL_HZ

MAX_TILT = 0.5  # rad
HOVER_ALTITUDE = 1.0
TRACKING_ALTITUDE = 0.5
TARGET_SPEED = 1.0  # m/s

CARDBOARD_HALF_EXTENTS = np.array([0.18, 0.14, 0.02])
CARDBOARD_COLOR = np.array([0.9, 0.35, 0.2, 0.9])
TARGET_COLOR = np.array([0.15, 0.8, 0.2, 1.0])
DRONE_HALF_HEIGHT = 0.03
COLLISION_MARGIN = 0.02

FPS = 30

TUNABLE_PARAMS = [
    ("Qp",  2.0,   1.0,  100.0),
    ("Qv",  0.5,   0.1,   20.0),
    ("Qf", 20.0,  10.0,  500.0),
    ("R",   0.5,  0.01,   20.0),
    ("a",   0.5,   1.0,   15.0),
    ("v",   0.25,  0.5,    5.0),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def cf_to_mpc_state(pos, vel):
    """Crazyflow (x, y, z) z-up -> MPC [px, vx, py, vy, pz, vz]."""
    return np.array([
        pos[0], vel[0],
        pos[2], vel[2],
        pos[1], vel[1],
    ])


def cf_pos_to_mpc(pos):
    """Crazyflow [x, y, z] -> MPC [px, py, pz]."""
    return [float(pos[0]), float(pos[2]), float(pos[1])]


def cf_vel_to_mpc(vel):
    """Crazyflow [vx, vy, vz] -> MPC [vx, vy, vz]."""
    return [float(vel[0]), float(vel[2]), float(vel[1])]


def mpc_pos_to_cf(pos):
    """MPC [px, py, pz] -> Crazyflow [x, y, z]."""
    return [float(pos[0]), float(pos[2]), float(pos[1])]


def mpc_accel_to_attitude(ax, ay, az, mass, thrust_max):
    """MPC accelerations -> Crazyflow attitude command [roll, pitch, yaw, thrust]."""
    pitch = np.clip(ax / G, -MAX_TILT, MAX_TILT)
    roll = np.clip(-az / G, -MAX_TILT, MAX_TILT)
    thrust = np.clip(mass * ay, 0.0, thrust_max)
    return np.array([roll, pitch, 0.0, thrust])


def draw_box(sim, pos, half_extents, rgba, yaw=0.0):
    """Draw a flat box marker in the Crazyflow viewer."""
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
# Runtime MPC tuning
# ---------------------------------------------------------------------------
class ParamTuner:
    def __init__(self, config: MPCConfig):
        self._config = config
        self._lock = threading.Lock()
        self._pending = False
        self._selected = 0

    def _get_value(self, idx):
        c = self._config
        if idx == 0:
            return c.Q_diag[0]
        if idx == 1:
            return c.Q_diag[1]
        if idx == 2:
            return c.Qf_diag[0]
        if idx == 3:
            return c.R_diag[0]
        if idx == 4:
            return c.a_max
        return c.v_max

    def _set_value(self, idx, val):
        c = self._config
        if idx == 0:
            c.Q_diag = [val, c.Q_diag[1], val, c.Q_diag[3], val, c.Q_diag[5]]
        elif idx == 1:
            c.Q_diag[1] = val
            c.Q_diag[3] = val
            c.Q_diag[5] = val
        elif idx == 2:
            c.Qf_diag = [val, val / 10, val, val / 10, val, val / 10]
        elif idx == 3:
            c.R_diag = [val, val, val]
        elif idx == 4:
            c.a_max = val
        elif idx == 5:
            c.v_max = val

    def select(self, index):
        with self._lock:
            if 0 <= index < len(TUNABLE_PARAMS):
                self._selected = index

    def adjust(self, direction):
        with self._lock:
            _, step, lo, hi = TUNABLE_PARAMS[self._selected]
            val = self._get_value(self._selected)
            self._set_value(self._selected, max(lo, min(hi, val + direction * step)))
            self._pending = True

    def maybe_rebuild(self, mpc):
        with self._lock:
            if not self._pending:
                return mpc
            self._pending = False
            config = self._config
        new_mpc = MPCController(config)
        new_mpc._d_hat = mpc._d_hat
        return new_mpc

    def status_line(self):
        with self._lock:
            parts = []
            for i, (name, _, _, _) in enumerate(TUNABLE_PARAMS):
                value = self._get_value(i)
                item = f"{name}={value:.4g}"
                parts.append(f">{item}<" if i == self._selected else item)
            return " ".join(parts)


# ---------------------------------------------------------------------------
# Keyboard input
# ---------------------------------------------------------------------------
class KeyState:
    def __init__(self):
        self._pressed = set()
        self._lock = threading.Lock()
        self.exit_flag = False
        self.started = False
        self.mode_request = None

    def on_press(self, key):
        try:
            k = key.char.lower()
        except AttributeError:
            k = key
        with self._lock:
            self._pressed.add(k)

        if key == keyboard.Key.space:
            self.started = True
        elif key == keyboard.Key.esc:
            self.exit_flag = True
        elif key == keyboard.Key.up:
            self.mode_request = ("tune", +1)
        elif key == keyboard.Key.down:
            self.mode_request = ("tune", -1)
        elif isinstance(k, str):
            if k == "m":
                self.mode_request = "manual_track"
            elif k == "t":
                self.mode_request = "cardboard_track"
            elif k == "l":
                self.mode_request = "land"
            elif k == "o":
                self.mode_request = "respawn_cardboard"
            elif k == "h":
                self.mode_request = "hover"
            elif k in "123456":
                self.mode_request = ("select", int(k) - 1)

    def on_release(self, key):
        try:
            k = key.char.lower()
        except AttributeError:
            k = key
        with self._lock:
            self._pressed.discard(k)

    def is_pressed(self, char):
        with self._lock:
            return char in self._pressed

    def pop_request(self):
        req = self.mode_request
        self.mode_request = None
        return req


# ---------------------------------------------------------------------------
# Manual target and landing cardboard
# ---------------------------------------------------------------------------
class ManualTarget:
    """Keyboard-driven point target on the ground in Crazyflow coordinates."""

    def __init__(self):
        self.pos = np.array([0.0, 0.0, 0.0])  # CF [x, y, z]
        self.vel = np.array([0.0, 0.0, 0.0])

    def reset(self):
        self.pos[:] = 0.0
        self.vel[:] = 0.0

    def update(self, keys, dt):
        vx = 0.0
        vy = 0.0
        if keys.is_pressed("w"):
            vx += TARGET_SPEED
        if keys.is_pressed("s"):
            vx -= TARGET_SPEED
        if keys.is_pressed("a"):
            vy += TARGET_SPEED
        if keys.is_pressed("d"):
            vy -= TARGET_SPEED

        self.vel[:] = [vx, vy, 0.0]
        self.pos += self.vel * dt
        self.pos[2] = 0.0

        return {
            "pos": cf_pos_to_mpc(self.pos),
            "vel": cf_vel_to_mpc(self.vel),
        }


def build_tracking_reference(target_state, N, dt, altitude):
    """Reference for following a manual point at fixed altitude."""
    ref = np.zeros((N + 1, 6))
    target_pos = np.array(target_state["pos"])
    target_vel = np.array(target_state["vel"])
    for k in range(N + 1):
        t = k * dt
        pred_pos = target_pos + target_vel * t
        ref[k, 0] = pred_pos[0]
        ref[k, 1] = target_vel[0]
        ref[k, 2] = altitude
        ref[k, 3] = 0.0
        ref[k, 4] = pred_pos[2]
        ref[k, 5] = target_vel[2]
    return ref


def random_cardboard_position(rng):
    return np.array([
        rng.uniform(-0.7, 0.7),
        rng.uniform(-0.7, 0.7),
        CARDBOARD_HALF_EXTENTS[2],
    ])


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
    thrust_max = float(sim.data.controls.attitude.params.thrust_max) * 4

    config = MPCConfig(dt=CONTROL_DT, horizon=25)
    mpc = MPCController(config)
    tuner = ParamTuner(config)
    N = config.horizon

    rng = np.random.default_rng()
    cardboard_cf = random_cardboard_position(rng)
    cardboard_state = {
        "pos": cf_pos_to_mpc(cardboard_cf),
        "vel": [0.0, 0.0, 0.0],
    }

    manual_target = ManualTarget()

    keys = KeyState()
    listener = keyboard.Listener(on_press=keys.on_press, on_release=keys.on_release)
    listener.start()

    READY = "ready"
    MANUAL_TRACK = "manual_track"
    CARDBOARD_TRACK = "cardboard_track"
    LAND = "land"
    LANDED = "landed"
    HOVER = "hover"
    phase = READY
    hover_target = [0.0, HOVER_ALTITUDE, 0.0]

    print("\n=== MPC Teleop Landing (Crazyflow) ===")
    print("Space starts takeoff.")
    print("W/S/A/D move the manual tracking point on the ground.")
    print("M tracks manual point, T tracks cardboard, L lands on cardboard, O respawns cardboard.")
    print("H hover, 1-6 + Up/Down tune, Esc quit.")
    print(
        "Cardboard spawn (CF) = "
        f"({cardboard_cf[0]:+.2f}, {cardboard_cf[1]:+.2f}, {cardboard_cf[2]:+.2f})"
    )
    print("=" * 72)

    step_count = 0
    try:
        while not keys.exit_flag:
            req = keys.pop_request()
            if req == "manual_track" and phase != READY:
                phase = MANUAL_TRACK
                print("\n>>> Mode: MANUAL TRACK")
            elif req == "cardboard_track" and phase != READY:
                phase = CARDBOARD_TRACK
                print("\n>>> Mode: CARDBOARD TRACK")
            elif req == "land" and phase != READY:
                phase = LAND
                print("\n>>> Mode: LAND")
            elif req == "respawn_cardboard":
                cardboard_cf = random_cardboard_position(rng)
                cardboard_state = {
                    "pos": cf_pos_to_mpc(cardboard_cf),
                    "vel": [0.0, 0.0, 0.0],
                }
                print(
                    "\n>>> Cardboard respawned at "
                    f"({cardboard_cf[0]:+.2f}, {cardboard_cf[1]:+.2f}, {cardboard_cf[2]:+.2f})"
                )
            elif req == "hover" and phase != READY:
                phase = HOVER
                manual_target.reset()
                print("\n>>> Mode: HOVER")
            elif isinstance(req, tuple) and req[0] == "select":
                tuner.select(req[1])
            elif isinstance(req, tuple) and req[0] == "tune":
                tuner.adjust(req[1])

            if phase == READY and keys.started:
                phase = MANUAL_TRACK
                print("\n>>> Starting MANUAL TRACK mode")

            pos = np.array(sim.data.states.pos[0, 0])
            vel = np.array(sim.data.states.vel[0, 0])
            x0 = cf_to_mpc_state(pos, vel)
            manual_target_state = manual_target.update(keys, CONTROL_DT)

            if phase == READY:
                cmd = np.zeros((sim.n_worlds, sim.n_drones, 4))
                active_target = hover_target
                ax = ay = az = 0.0
                attitude = cmd[0, 0, :]
            elif phase == LANDED:
                cmd = np.zeros((sim.n_worlds, sim.n_drones, 4))
                active_target = cardboard_state["pos"]
                ax = ay = az = 0.0
                attitude = cmd[0, 0, :]
            else:
                mpc = tuner.maybe_rebuild(mpc)
                drone_state = {
                    "pos": [x0[0], x0[2], x0[4]],
                    "vel": [x0[1], x0[3], x0[5]],
                }
                if phase == MANUAL_TRACK:
                    ref = build_tracking_reference(
                        manual_target_state,
                        N,
                        CONTROL_DT,
                        TRACKING_ALTITUDE,
                    )
                elif phase == CARDBOARD_TRACK:
                    ref = build_tracking_reference(
                        cardboard_state,
                        N,
                        CONTROL_DT,
                        TRACKING_ALTITUDE,
                    )
                elif phase == LAND:
                    ref = landing_reference(drone_state, cardboard_state, N, CONTROL_DT)
                else:
                    ref = static_reference(hover_target, N, CONTROL_DT)

                active_target = [float(ref[0, 0]), float(ref[0, 2]), float(ref[0, 4])]
                u_opt = mpc.compute(x0, ref)
                ax, ay, az = u_opt
                attitude = mpc_accel_to_attitude(ax, ay, az, mass, thrust_max)
                cmd = np.zeros((sim.n_worlds, sim.n_drones, 4))
                cmd[0, 0, :] = attitude

            sim.attitude_control(cmd)
            sim.step(STEPS_PER_MPC)

            drone_pos = np.array(sim.data.states.pos[0, 0])
            platform_top = 2 * CARDBOARD_HALF_EXTENTS[2] + DRONE_HALF_HEIGHT
            in_footprint = (
                abs(drone_pos[0] - cardboard_cf[0]) < CARDBOARD_HALF_EXTENTS[0] + COLLISION_MARGIN
                and abs(drone_pos[1] - cardboard_cf[1]) < CARDBOARD_HALF_EXTENTS[1] + COLLISION_MARGIN
            )
            if phase == LAND and in_footprint and drone_pos[2] < platform_top:
                new_pos = sim.data.states.pos.at[0, 0, 2].set(platform_top)
                new_vel = sim.data.states.vel
                if float(sim.data.states.vel[0, 0, 2]) < 0:
                    new_vel = new_vel.at[0, 0, 2].set(0.0)
                sim.data = sim.data.replace(
                    states=sim.data.states.replace(pos=new_pos, vel=new_vel)
                )
                phase = LANDED
                print("\n>>> LANDED -- press H, M, or T to fly again")

            step_count += 1
            if (step_count * FPS) % CONTROL_HZ < FPS:
                draw_box(sim, cardboard_cf, CARDBOARD_HALF_EXTENTS, CARDBOARD_COLOR)
                target_cf = np.array([manual_target.pos])
                draw_points(sim, target_cf, rgba=TARGET_COLOR, size=0.04)
                sim.render()

            if step_count % 5 == 0:
                active_target_cf = mpc_pos_to_cf(active_target)
                status = (
                    f"\r[{phase:>6s}] "
                    f"drone=({pos[0]:+.2f},{pos[1]:+.2f},{pos[2]:+.2f}) "
                    f"target=({manual_target.pos[0]:+.2f},{manual_target.pos[1]:+.2f},{manual_target.pos[2]:+.2f}) "
                    f"cardboard=({cardboard_cf[0]:+.2f},{cardboard_cf[1]:+.2f},{cardboard_cf[2]:+.2f}) "
                    f"ref=({active_target_cf[0]:+.2f},{active_target_cf[1]:+.2f},{active_target_cf[2]:+.2f}) "
                    f"a=({ax:+.2f},{ay:+.2f},{az:+.2f}) "
                    f"T={attitude[3]:.3f} | {tuner.status_line()}  "
                )
                print(status, end="")

    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        listener.stop()
        sim.close()
        print("\nDone.")


if __name__ == "__main__":
    main()
