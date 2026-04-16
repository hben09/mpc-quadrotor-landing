"""
Crazyflow simulation counterpart for mpc_teleop_landing.py.

Mirrors the hardware script's mode logic:
- Manual mode (default after takeoff):
    WASD move target in XZ, Z/X adjust altitude, Q/E adjust target yaw.
- Tracking mode (toggle with T):
    Drone tracks the movable landing object at 1.0 m above it.
    Q/E are ignored while tracking is enabled.
- Landing mode (toggle with L):
    Drone descends onto the landing object at 0.3 m/s.
    Motors auto-cut when within TOUCHDOWN_MARGIN of the pad.

Switching out of tracking/landing back to manual keeps the drone near its
current reference (no snap back to the previous manual target), matching the
hardware script behavior.

Controls:
    Space  : start takeoff / control loop
    W/S/A/D: manual target motion in manual mode; move landing pad in T/L mode
    Z/X    : lower / raise target altitude (manual only)
    Q/E    : target yaw (manual only)
    T      : toggle tracking mode
    L      : toggle landing mode
    1-6    : select MPC parameter
    Up/Down: tune selected MPC parameter
    Esc    : quit

Usage:
    uv run python simu_mpc_teleop_landing.py
"""

import csv
import threading
import time
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np
from pynput import keyboard

from crazyflow.control import Control
from crazyflow.sim import Physics, Sim
from crazyflow.sim.integration import Integrator
from crazyflow.sim.visualize import draw_points

from mpc_landing import MPCController, MPCConfig
from mpc_landing.reference import landing_reference, static_reference

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
TRACKING_ALTITUDE_OFFSET = 1.0
TARGET_SPEED = 0.5  # m/s, match hardware script
YAW_SPEED = np.radians(60.0)
LANDING_DESCENT_RATE = 0.3
TOUCHDOWN_MARGIN = 0.10

LANDING_HALF_EXTENTS = np.array([0.18, 0.14, 0.02])
LANDING_COLOR = np.array([0.90, 0.35, 0.20, 0.90])
MANUAL_TARGET_COLOR = np.array([0.15, 0.80, 0.20, 1.00])
TRACK_TARGET_COLOR = np.array([0.20, 0.55, 1.00, 1.00])
PLANNED_COLOR = np.array([1.00, 0.95, 0.20, 0.80])
FPS = 30

TARGET = [0.0, HOVER_ALTITUDE, 0.0]  # MPC coords [x, y, z]
TARGET_YAW = 0.0

TUNABLE_PARAMS = [
    ("Qp", 2.0, 1.0, 100.0),
    ("Qv", 0.5, 0.1, 20.0),
    ("Qf", 20.0, 10.0, 500.0),
    ("R", 0.5, 0.01, 20.0),
    ("a", 0.5, 1.0, 15.0),
    ("v", 0.25, 0.5, 5.0),
]

_LOG_COLUMNS = [
    "t",
    "px",
    "py",
    "pz",
    "vx",
    "vy",
    "vz",
    "tx",
    "ty",
    "tz",
    "ax",
    "ay",
    "az",
    "roll",
    "pitch",
    "thrust",
    "d_hat",
    "yaw",
    "target_yaw",
    "yawrate",
    "Qp",
    "Qv",
    "Qf",
    "R",
    "a_max",
    "v_max",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def wrap_to_pi(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


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


def rigid_body_like_state_from_mpc(x0):
    return {
        "pos": [float(x0[0]), float(x0[2]), float(x0[4])],
        "vel": [float(x0[1]), float(x0[3]), float(x0[5])],
    }


def tracking_reference_sim(target_state, horizon, dt, altitude_offset=1.0):
    """Tracking reference matching hardware intent: hold fixed altitude above pad."""
    ref = np.zeros((horizon + 1, 6))
    target_pos = np.array(target_state["pos"], dtype=float)
    target_vel = np.array(target_state["vel"], dtype=float)
    tracking_altitude = target_pos[1] + altitude_offset

    for k in range(horizon + 1):
        t = k * dt
        pred_pos = target_pos + target_vel * t
        ref[k, 0] = pred_pos[0]
        ref[k, 1] = target_vel[0]
        ref[k, 2] = tracking_altitude
        ref[k, 3] = 0.0
        ref[k, 4] = pred_pos[2]
        ref[k, 5] = target_vel[2]
    return ref


def mpc_accel_to_attitude(ax, ay, az, mass, thrust_max, yaw=0.0):
    """MPC accelerations -> Crazyflow attitude command [roll, pitch, yaw, thrust]."""
    pitch = np.clip(ax / G, -MAX_TILT, MAX_TILT)
    roll = np.clip(-az / G, -MAX_TILT, MAX_TILT)
    thrust = np.clip(mass * ay, 0.0, thrust_max)
    return np.array([roll, pitch, yaw, thrust])


def read_sim_yaw(sim):
    """Best-effort yaw readback from sim state, or None if unavailable."""
    for attr in ("rpy", "euler"):
        value = getattr(sim.data.states, attr, None)
        if value is None:
            continue
        angles = np.asarray(value[0, 0], dtype=float)
        if angles.shape[-1] >= 3:
            return float(angles[2])
    return None


def draw_box(sim, pos, half_extents, rgba, yaw=0.0):
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


class TeleopLogger:
    """CSV logger using the hardware teleop schema plus a mode column.

    The column names match hardware/mpc_teleop_landing.py. In sim, the
    setpoint values are simulator units: roll/pitch in radians and thrust in N.
    """

    def __init__(self, log_dir: Path):
        log_dir.mkdir(exist_ok=True)
        self.path = log_dir / f"teleop_{datetime.now():%Y%m%d_%H%M%S}.csv"
        self._file = None
        self._writer = None

    def __enter__(self):
        self._file = open(self.path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow([*_LOG_COLUMNS, "mode"])
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._file is not None:
            self._file.close()

    def log(
        self,
        *,
        t,
        pos,
        vel,
        target,
        accel,
        setpoint,
        disturbance,
        yaw,
        target_yaw,
        yawrate,
        config,
        mode,
    ):
        roll, pitch, thrust = setpoint
        self._writer.writerow([
            f"{t:.3f}",
            f"{pos[0]:.4f}",
            f"{pos[1]:.4f}",
            f"{pos[2]:.4f}",
            f"{vel[0]:.4f}",
            f"{vel[1]:.4f}",
            f"{vel[2]:.4f}",
            f"{target[0]:.4f}",
            f"{target[1]:.4f}",
            f"{target[2]:.4f}",
            f"{accel[0]:.4f}",
            f"{accel[1]:.4f}",
            f"{accel[2]:.4f}",
            f"{roll:.4f}",
            f"{pitch:.4f}",
            f"{thrust:.4f}",
            f"{disturbance:.4f}",
            f"{yaw:.4f}",
            f"{target_yaw:.4f}",
            f"{yawrate:.2f}",
            f"{config.Q_diag[0]:.2f}",
            f"{config.Q_diag[1]:.2f}",
            f"{config.Qf_diag[0]:.2f}",
            f"{config.R_diag[0]:.2f}",
            f"{config.a_max:.2f}",
            f"{config.v_max:.2f}",
            mode,
        ])


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
# Keyboard input and movable targets
# ---------------------------------------------------------------------------
class KeyState:
    def __init__(self):
        self._pressed = set()
        self._lock = threading.Lock()
        self.exit_flag = False
        self.started = False
        self.toggle_tracking = False
        self.toggle_landing = False
        self.tune_request = None
        self.select_request = None

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
            self.tune_request = +1
        elif key == keyboard.Key.down:
            self.tune_request = -1
        elif isinstance(k, str):
            if k == "t":
                self.toggle_tracking = True
            elif k == "l":
                self.toggle_landing = True
            elif k in "123456":
                self.select_request = int(k) - 1

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

    def pop_toggle_tracking(self):
        out = self.toggle_tracking
        self.toggle_tracking = False
        return out

    def pop_toggle_landing(self):
        out = self.toggle_landing
        self.toggle_landing = False
        return out

    def pop_tune_request(self):
        out = self.tune_request
        self.tune_request = None
        return out

    def pop_select_request(self):
        out = self.select_request
        self.select_request = None
        return out


class ManualTarget:
    def update(self, keys, dt):
        global TARGET, TARGET_YAW
        step = TARGET_SPEED * dt
        if keys.is_pressed("w"):
            TARGET[0] += step
        if keys.is_pressed("s"):
            TARGET[0] -= step
        if keys.is_pressed("d"):
            TARGET[2] += step
        if keys.is_pressed("a"):
            TARGET[2] -= step
        if keys.is_pressed("x"):
            TARGET[1] = min(TARGET[1] + step, 2.0)
        if keys.is_pressed("z"):
            TARGET[1] = max(TARGET[1] - step, 0.3)
        if keys.is_pressed("q"):
            TARGET_YAW += YAW_SPEED * dt
        if keys.is_pressed("e"):
            TARGET_YAW -= YAW_SPEED * dt
        TARGET_YAW = wrap_to_pi(TARGET_YAW)
        return list(TARGET), TARGET_YAW


class LandingPad:
    def __init__(self):
        self.pos = np.array([0.0, 0.0, LANDING_HALF_EXTENTS[2]], dtype=float)  # CF coords
        self.vel = np.zeros(3, dtype=float)
        self.yaw = 0.0

    def update(self, keys, dt, movable):
        vx = 0.0
        vy = 0.0
        if movable:
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
        self.pos[2] = LANDING_HALF_EXTENTS[2]
        return {
            "pos": cf_pos_to_mpc(self.pos),
            "vel": cf_vel_to_mpc(self.vel),
            "yaw": self.yaw,
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global TARGET_YAW, TARGET

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
    horizon = config.horizon

    keys = KeyState()
    listener = keyboard.Listener(on_press=keys.on_press, on_release=keys.on_release)
    listener.start()

    manual_target = ManualTarget()
    landing_pad = LandingPad()

    tracking_enabled = False
    landing_enabled = False
    started = False
    prev_mode = "M"
    active_target = list(TARGET)
    log_dir = Path(__file__).resolve().parent / "logs"

    print()
    print(f"=== MPC Teleop + Tracking + Landing (Sim) — target {TARGET} ===")
    print("Press SPACE to take off, Esc to abort")
    print("Manual:   WASD move XZ, Z/X altitude, Q/E yaw target")
    print("Tracking: T toggles tracking, drone holds 1.0 m above landing pad")
    print("Landing:  L toggles descent onto landing pad")
    print("In tracking/landing mode, WASD moves the landing pad")
    print("MPC tuning: 1-6 select, Up/Down adjust")
    print("  1:Q_pos 2:Q_vel 3:Qf 4:R 5:a_max 6:v_max")
    print("=" * 72)

    step_count = 0
    with TeleopLogger(log_dir) as log:
        t0_mpc = None
        prev_logged_yaw = None
        log_announced = False

        try:
            while not keys.exit_flag:
                if keys.pop_toggle_tracking() and started:
                    if tracking_enabled:
                        tracking_enabled = False
                        print("\n>>> Tracking OFF (manual)")
                    else:
                        landing_enabled = False
                        tracking_enabled = True
                        print("\n>>> Tracking ON (landing)")

                if keys.pop_toggle_landing() and started:
                    if landing_enabled:
                        landing_enabled = False
                        print("\n>>> Landing OFF (manual)")
                    else:
                        tracking_enabled = False
                        landing_enabled = True
                        print("\n>>> Landing ON (descending onto landing)")

                select_request = keys.pop_select_request()
                if select_request is not None:
                    tuner.select(select_request)

                tune_request = keys.pop_tune_request()
                if tune_request is not None:
                    tuner.adjust(tune_request)

                if not started and keys.started:
                    started = True
                    t0_mpc = time.monotonic()
                    prev_logged_yaw = None
                    if not log_announced:
                        print(f"\nLogging CSV to {log.path}")
                        log_announced = True
                    print("\n>>> Starting control loop")

                pos = np.array(sim.data.states.pos[0, 0])
                vel = np.array(sim.data.states.vel[0, 0])
                x0 = cf_to_mpc_state(pos, vel)

                # Move landing pad whenever autonomous pad-based mode is active.
                pad_state = landing_pad.update(
                    keys,
                    CONTROL_DT,
                    movable=(tracking_enabled or landing_enabled),
                )

                mode = "R"
                ax = ay = az = 0.0
                attitude = np.zeros(4)

                if started:
                    mpc = tuner.maybe_rebuild(mpc)

                    if landing_enabled:
                        drone_state = rigid_body_like_state_from_mpc(x0)
                        ref = landing_reference(
                            drone_state,
                            {"pos": pad_state["pos"], "vel": pad_state["vel"]},
                            horizon,
                            CONTROL_DT,
                            descent_rate=LANDING_DESCENT_RATE,
                        )
                        active_target = [
                            float(ref[0, 0]),
                            float(ref[0, 2]),
                            float(ref[0, 4]),
                        ]
                        TARGET_YAW = pad_state["yaw"]
                        mode = "L"

                        pad_height = pad_state["pos"][1] + 0.05
                        if x0[2] <= pad_height + TOUCHDOWN_MARGIN:
                            cmd = np.zeros((sim.n_worlds, sim.n_drones, 4))
                            sim.attitude_control(cmd)
                            sim.step(STEPS_PER_MPC)
                            landing_enabled = False
                            started = False
                            TARGET = list(active_target)
                            print(
                                f"\n*** TOUCHDOWN at y={x0[2]:.2f} m — motors cut ***"
                            )
                            continue

                    elif tracking_enabled:
                        ref = tracking_reference_sim(
                            {"pos": pad_state["pos"], "vel": pad_state["vel"]},
                            horizon,
                            CONTROL_DT,
                            altitude_offset=TRACKING_ALTITUDE_OFFSET,
                        )
                        active_target = [
                            float(ref[0, 0]),
                            float(ref[0, 2]),
                            float(ref[0, 4]),
                        ]
                        TARGET_YAW = pad_state["yaw"]
                        mode = "T"
                    else:
                        if prev_mode in ("T", "L"):
                            TARGET = list(active_target)
                        active_target, TARGET_YAW = manual_target.update(
                            keys,
                            CONTROL_DT,
                        )
                        ref = static_reference(TARGET, horizon, CONTROL_DT)
                        mode = "M"

                    prev_mode = mode

                    u_opt = mpc.compute(x0, ref)
                    ax, ay, az = u_opt
                    attitude = mpc_accel_to_attitude(
                        ax,
                        ay,
                        az,
                        mass,
                        thrust_max,
                        yaw=TARGET_YAW,
                    )
                    cmd = np.zeros((sim.n_worlds, sim.n_drones, 4))
                    cmd[0, 0, :] = attitude

                    logged_yaw = read_sim_yaw(sim)
                    if logged_yaw is None:
                        logged_yaw = float(attitude[2])
                    if prev_logged_yaw is None:
                        yawrate_deg = 0.0
                    else:
                        yawrate_deg = np.degrees(
                            wrap_to_pi(logged_yaw - prev_logged_yaw)
                        ) / CONTROL_DT
                    prev_logged_yaw = logged_yaw

                    log.log(
                        t=(time.monotonic() - t0_mpc) if t0_mpc is not None else 0.0,
                        pos=[float(x0[0]), float(x0[2]), float(x0[4])],
                        vel=[float(x0[1]), float(x0[3]), float(x0[5])],
                        target=active_target,
                        accel=[float(ax), float(ay), float(az)],
                        setpoint=(
                            float(attitude[0]),
                            float(attitude[1]),
                            float(attitude[3]),
                        ),
                        disturbance=mpc.disturbance,
                        yaw=logged_yaw,
                        target_yaw=TARGET_YAW,
                        yawrate=yawrate_deg,
                        config=config,
                        mode=mode,
                    )
                else:
                    cmd = np.zeros((sim.n_worlds, sim.n_drones, 4))

                sim.attitude_control(cmd)
                sim.step(STEPS_PER_MPC)

                step_count += 1
                if (step_count * FPS) % CONTROL_HZ < FPS:
                    draw_box(
                        sim,
                        landing_pad.pos,
                        LANDING_HALF_EXTENTS,
                        LANDING_COLOR,
                        yaw=landing_pad.yaw,
                    )
                    draw_points(
                        sim,
                        np.array([mpc_pos_to_cf(TARGET)]),
                        rgba=MANUAL_TARGET_COLOR,
                        size=0.04,
                    )
                    draw_points(
                        sim,
                        np.array([
                            [
                                landing_pad.pos[0],
                                landing_pad.pos[1],
                                landing_pad.pos[2] + 0.03,
                            ]
                        ]),
                        rgba=TRACK_TARGET_COLOR,
                        size=0.04,
                    )
                    planned = mpc.get_planned_trajectory()
                    if planned is not None and started:
                        pts = np.array([
                            mpc_pos_to_cf([r[0], r[2], r[4]])
                            for r in planned
                        ])
                        draw_points(sim, pts, rgba=PLANNED_COLOR, size=0.012)
                    sim.render()

                if step_count % 5 == 0:
                    active_target_cf = mpc_pos_to_cf(active_target)
                    status = (
                        f"\r[{mode}] "
                        f"drone=({pos[0]:+.2f},{pos[1]:+.2f},{pos[2]:+.2f}) "
                        f"manual=({TARGET[0]:+.2f},{TARGET[2]:+.2f},{TARGET[1]:+.2f}) "
                        f"landing=({landing_pad.pos[0]:+.2f},{landing_pad.pos[1]:+.2f},{landing_pad.pos[2]:+.2f}) "
                        f"ref=({active_target_cf[0]:+.2f},{active_target_cf[1]:+.2f},{active_target_cf[2]:+.2f}) "
                        f"a=({ax:+.2f},{ay:+.2f},{az:+.2f}) "
                        f"yaw_tgt={np.degrees(TARGET_YAW):+5.1f}° "
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
