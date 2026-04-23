"""
MPC teleop + tracking + landing — fly via a manually-piloted setpoint, with
T toggling autonomous tracking of the OptiTrack rigid body published as
``rb/landing``, and L toggling autonomous descent onto it.

Manual mode (default):
    WASD move target in XZ, Z/X adjust altitude, Q/E adjust target yaw.
Tracking mode (toggle with T):
    Drone holds 1 m above the landing object and yaws to match its heading.
    Q/E are ignored until tracking is toggled off again.
Landing mode (toggle with L):
    Drone descends onto the landing object at 0.3 m/s and yaws to match its
    heading. Motors auto-cut when within TOUCHDOWN_MARGIN of the pad. After
    touchdown the process stays alive — press SPACE to take off again.

Switching out of tracking/landing back to manual keeps the drone near its
current reference (no snap back to the previous manual target).

Press SPACE to take off (or re-takeoff after landing), T/L to toggle
tracking/landing, Esc to stop.

Usage:
    uv run python hardware/mpc_pilot.py
"""

import json
import sys
import time
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
from pynput import keyboard

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils import uri_helper

import paho.mqtt.client as mqtt

from mpc_landing import MPCController, MPCConfig
from mpc_landing.boundary import ARENA_BOUNDS
from mpc_landing.mqtt.parser import RigidBodyTracker
from mpc_landing.guidance import (
    APPROACH_CONE_BASE_RADIUS_M,
    APPROACH_CONE_HALF_ANGLE_DEG,
    is_in_approach_cone,
    landing_reference,
    static_reference,
    tracking_reference,
)
from mpc_landing.supervisor import SafeCommander
from mpc_landing.yaw_controller import compute_yawrate, wrap_to_pi

from battery import BatteryPublisher
from csv_logger import TeleopLogger, InfeasibilityLogger, EventLogger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
G = 9.81
CONTROL_HZ = 50
CONTROL_DT = 1.0 / CONTROL_HZ

HOVER_PWM = 45000
HOVER_ALTITUDE = 1.0
MAX_TILT_DEG = 10.0
TARGET = [0.0, HOVER_ALTITUDE, 0.0]
TARGET_YAW = 0.0  # radians, OptiTrack frame (CCW+)
YAW_SPEED = np.radians(60.0)  # rad/s while Left/Right held

MQTT_BROKER = "rasticvm.internal"
MQTT_PORT = 1883
DRONE_TOPIC = "rb/crazyflie"
TRACKED_OBJECT_TOPIC = "rb/landing"
TRACKED_OBJECT_NAME = "landing"
MPC_TARGET_TOPIC = "mpc/target"
MPC_TRAJ_TOPIC = "mpc/trajectory"
MPC_REF_TOPIC = "mpc/reference"
MPC_CONE_TOPIC = "mpc/cone"
BATTERY_TOPIC = "cf/battery"

TARGET_SPEED = 0.5  # meters per second (WASD/QE, continuous while held)

RAMP_DURATION = 1.5
AIRBORNE_ALT = 0.3
MIN_POSE_COUNT = 3
TOUCHDOWN_MARGIN = 0.05  # meters above landing pad at which motors auto-cut
TOUCHDOWN_RAMP_DURATION = 0.5  # seconds to linearly ramp thrust to 0 after touchdown

# Pressed-key tracking for smooth target movement
pressed_keys = set()
keys_lock = threading.Lock()

# Tracking / landing mode toggles (mutually exclusive)
tracking_enabled = threading.Event()
landing_enabled = threading.Event()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def optitrack_to_mpc_state(rb):
    return np.array(
        [
            rb.pos[0],
            rb.vel[0],
            rb.pos[1],
            rb.vel[1],
            rb.pos[2],
            rb.vel[2],
        ]
    )


def mpc_accel_to_cflib_setpoint(ax, ay, az, yaw):
    # Rotate world-frame (ax, az) into body frame using drone yaw (OptiTrack
    # euler[1], + = CCW from above). cflib interprets pitch/roll in body frame.
    c, s = np.cos(yaw), np.sin(yaw)
    a_fwd = c * ax - s * az
    a_right = s * ax + c * az
    pitch_deg = float(np.clip(np.degrees(a_fwd / G), -MAX_TILT_DEG, MAX_TILT_DEG))
    roll_deg = float(np.clip(np.degrees(a_right / G), -MAX_TILT_DEG, MAX_TILT_DEG))
    thrust_pwm = int(np.clip(HOVER_PWM * (ay / G), 0, 60000))
    return roll_deg, pitch_deg, thrust_pwm


def rigid_body_to_state(rb):
    return {
        "pos": list(rb.pos),
        "vel": list(rb.vel),
        "yaw_rate": rb.yaw_rate,
    }


# ---------------------------------------------------------------------------
# Runtime MPC parameter tuner (keyboard-driven)
# ---------------------------------------------------------------------------
TUNABLE_PARAMS = [
    # (short_name, step, min, max, getter, setter)
    # getter/setter operate on an MPCConfig instance
    ("Qp", 2.0, 1.0, 100.0),  # 1: Q position
    ("Qv", 0.5, 0.1, 20.0),  # 2: Q velocity
    ("Qf", 20.0, 10.0, 500.0),  # 3: Qf terminal
    ("R", 0.5, 0.01, 20.0),  # 4: R effort
    ("a", 0.5, 1.0, 15.0),  # 5: a_max
    ("v", 0.25, 0.5, 5.0),  # 6: v_max
]


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
        elif idx == 1:
            return c.Q_diag[1]
        elif idx == 2:
            return c.Qf_diag[0]
        elif idx == 3:
            return c.R_diag[0]
        elif idx == 4:
            return c.a_max
        elif idx == 5:
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
            p = TUNABLE_PARAMS[self._selected]
            _, step, lo, hi = p
            val = self._get_value(self._selected)
            val = max(lo, min(hi, val + direction * step))
            self._set_value(self._selected, val)
            self._pending = True

    def maybe_rebuild(self, mpc):
        with self._lock:
            if not self._pending:
                return mpc
            self._pending = False
            config = self._config
        try:
            new_mpc = MPCController(config)
            new_mpc._d_hat = mpc._d_hat  # preserve disturbance estimate
            return new_mpc
        except Exception as e:
            print(f"\nMPC rebuild failed: {e}")
            return mpc

    def status_line(self):
        with self._lock:
            parts = []
            for i, (name, _, _, _) in enumerate(TUNABLE_PARAMS):
                val = self._get_value(i)
                txt = f"{val:.4g}"
                if i == self._selected:
                    parts.append(f">{name}={txt}<")
                else:
                    parts.append(f"{name}={txt}")
            return " ".join(parts)


# ---------------------------------------------------------------------------
# MQTT state reader (drone + tracked object)
# ---------------------------------------------------------------------------
class TrackingStateReader:
    def __init__(self):
        self._lock = threading.Lock()
        self._drone_tracker = RigidBodyTracker()
        self._target_tracker = RigidBodyTracker()
        self._drone = None
        self._target = None
        self._drone_count = 0
        self._target_count = 0
        self._client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id="mpc-pilot-reader",
            protocol=mqtt.MQTTv311,
        )
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_msg

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        client.subscribe(DRONE_TOPIC)
        client.subscribe(TRACKED_OBJECT_TOPIC)

    def _on_msg(self, client, userdata, msg):
        with self._lock:
            if msg.topic == DRONE_TOPIC:
                self._drone = self._drone_tracker.update(msg.payload.decode())
                self._drone_count += 1
            elif msg.topic == TRACKED_OBJECT_TOPIC:
                self._target = self._target_tracker.update(msg.payload.decode())
                self._target_count += 1

    def get_drone(self):
        with self._lock:
            return self._drone

    def get_target(self):
        with self._lock:
            return self._target

    def drone_count(self):
        with self._lock:
            return self._drone_count

    def target_count(self):
        with self._lock:
            return self._target_count

    def start(self):
        self._client.connect(MQTT_BROKER, MQTT_PORT)
        self._client.loop_start()

    def stop(self):
        self._client.loop_stop()
        self._client.disconnect()


def on_release(key):
    with keys_lock:
        try:
            pressed_keys.discard(key.char.lower())
        except AttributeError:
            pressed_keys.discard(key)


def update_target(dt):
    """Move TARGET (and TARGET_YAW) continuously based on held keys."""
    global TARGET_YAW
    with keys_lock:
        keys = set(pressed_keys)
    step = TARGET_SPEED * dt
    if "w" in keys:
        TARGET[0] += step
    if "s" in keys:
        TARGET[0] -= step
    if "d" in keys:
        TARGET[2] += step
    if "a" in keys:
        TARGET[2] -= step
    if "x" in keys:
        TARGET[1] = min(TARGET[1] + step, 2.0)
    if "z" in keys:
        TARGET[1] = max(TARGET[1] - step, 0.3)
    if "q" in keys:
        TARGET_YAW += YAW_SPEED * dt  # q = left = CCW = + in OptiTrack
    if "e" in keys:
        TARGET_YAW -= YAW_SPEED * dt  # e = right = CW
    TARGET_YAW = wrap_to_pi(TARGET_YAW)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global TARGET_YAW

    cflib.crtp.init_drivers()

    print("Scanning for Crazyflie...")
    available = cflib.crtp.scan_interfaces()
    if not available:
        print("No Crazyflie found!")
        sys.exit(1)

    uri = available[0][0]
    print(f"Found: {uri}")
    print("Connecting...")

    cache_dir = str(Path(__file__).resolve().parent.parent / "cache")
    with SyncCrazyflie(uri, cf=Crazyflie(rw_cache=cache_dir)) as scf:
        cf = scf.cf
        cf.platform.send_arming_request(True)
        time.sleep(1.0)
        cf.commander.send_setpoint(0, 0, 0, 0)

        # MPC
        config = MPCConfig(dt=CONTROL_DT, horizon=50)
        mpc = MPCController(config)
        tuner = ParamTuner(config)
        N = config.horizon

        # MQTT
        reader = TrackingStateReader()
        reader.start()

        pub = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id="mpc-pilot-pub",
            protocol=mqtt.MQTTv311,
        )
        pub.connect(MQTT_BROKER, MQTT_PORT)
        pub.loop_start()

        battery = BatteryPublisher(cf, pub)
        battery.start()

        # Subscribe to the same battery topic so the MPC loop can log vbat.
        battery_state = {"vbat": None}
        battery_lock = threading.Lock()

        def _on_battery(_client, _userdata, msg):
            try:
                data = json.loads(msg.payload.decode())
                with battery_lock:
                    battery_state["vbat"] = float(data.get("vbat", 0.0))
            except Exception:
                pass

        pub.message_callback_add(BATTERY_TOPIC, _on_battery)
        pub.subscribe(BATTERY_TOPIC)

        # Wait for OptiTrack drone pose (landing pose is optional until T is pressed)
        print("Waiting for OptiTrack...", end="", flush=True)
        t0 = time.monotonic()
        while reader.drone_count() < MIN_POSE_COUNT:
            if time.monotonic() - t0 > 10.0:
                print(" TIMEOUT")
                reader.stop()
                pub.loop_stop()
                sys.exit(1)
            time.sleep(0.05)
        print(" OK")

        # Publish target so mqtt_viewer shows it
        pub.publish(MPC_TARGET_TOPIC, json.dumps({"pos": TARGET, "yaw": TARGET_YAW}))

        # Wait for SPACE
        go = threading.Event()
        stop = threading.Event()
        landing_missing_warned = [False]

        def on_press(key):
            if key == keyboard.Key.space:
                go.set()
            if key == keyboard.Key.esc:
                stop.set()
                go.set()  # unblock wait
            if key == keyboard.Key.up:
                tuner.adjust(+1)
            if key == keyboard.Key.down:
                tuner.adjust(-1)
            # Track held keys for smooth target movement and one-shot toggles
            with keys_lock:
                try:
                    if hasattr(key, "char") and key.char:
                        c = key.char.lower()
                        is_new_press = c not in pressed_keys
                        if c in "123456":
                            tuner.select(int(c) - 1)
                        if c == "t" and is_new_press:
                            if tracking_enabled.is_set():
                                tracking_enabled.clear()
                                print("\n>>> Tracking OFF (manual)")
                            else:
                                if reader.get_target() is None:
                                    if not landing_missing_warned[0]:
                                        print(
                                            f"\n>>> No '{TRACKED_OBJECT_NAME}' pose yet"
                                            f" — staying manual"
                                        )
                                        landing_missing_warned[0] = True
                                else:
                                    landing_enabled.clear()
                                    tracking_enabled.set()
                                    print(f"\n>>> Tracking ON ({TRACKED_OBJECT_NAME})")
                        if c == "l" and is_new_press:
                            if landing_enabled.is_set():
                                landing_enabled.clear()
                                print("\n>>> Landing OFF (manual)")
                            else:
                                if reader.get_target() is None:
                                    if not landing_missing_warned[0]:
                                        print(
                                            f"\n>>> No '{TRACKED_OBJECT_NAME}' pose yet"
                                            f" — staying manual"
                                        )
                                        landing_missing_warned[0] = True
                                else:
                                    tracking_enabled.clear()
                                    landing_enabled.set()
                                    print(
                                        f"\n>>> Landing ON (descending onto '{TRACKED_OBJECT_NAME}')"
                                    )
                        pressed_keys.add(c)
                except AttributeError:
                    pressed_keys.add(key)

        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()

        print()
        print(f"=== MPC Teleop + Tracking + Landing — target {TARGET} ===")
        print("Press SPACE to take off (or re-takeoff after landing), Esc to abort")
        print("WASD = move XZ, Z/X = altitude, Q/E = yaw target (hold)")
        print(f"T = toggle tracking of OptiTrack rigid body '{TRACKED_OBJECT_NAME}'")
        print(f"L = toggle autonomous descent onto '{TRACKED_OBJECT_NAME}'")
        print("MPC tuning: 1-6 select, Up/Down adjust")
        print("  1:Q_pos 2:Q_vel 3:Qf 4:R 5:a_max 6:v_max")
        print("=" * 56)

        go.wait()
        if stop.is_set():
            print("Aborted.")
            battery.stop()
            reader.stop()
            pub.loop_stop()
            pub.disconnect()
            listener.stop()
            return

        with SafeCommander(cf.commander) as commander:
            try:
                flight_dir = (
                    Path(__file__).resolve().parent
                    / "logs"
                    / f"teleop_{datetime.now():%Y%m%d_%H%M%S}"
                )
                with (
                    TeleopLogger(flight_dir, include_mode=True) as log,
                    InfeasibilityLogger(flight_dir) as infeas,
                    EventLogger(flight_dir) as events,
                ):
                    # Flight loop: one iteration per takeoff→landing cycle.
                    # t0 is set once so log timestamps are session-wide.
                    t0 = time.monotonic()
                    while not stop.is_set():
                        # Takeoff ramp
                        print("Takeoff ramp...")
                        ramp_start = time.monotonic()
                        events.log(
                            time.monotonic() - t0,
                            "takeoff_start",
                            hover_pwm=HOVER_PWM,
                            duration_s=RAMP_DURATION,
                        )
                        while not stop.is_set():
                            elapsed = time.monotonic() - ramp_start
                            if elapsed >= RAMP_DURATION:
                                break
                            if commander.boundary_violated:
                                events.log(
                                    time.monotonic() - t0,
                                    "boundary_violated",
                                    phase="takeoff",
                                )
                                break
                            frac = elapsed / RAMP_DURATION
                            commander.send_setpoint(
                                0.0, 0.0, 0.0, int(HOVER_PWM * frac)
                            )
                            drone = reader.get_drone()
                            if drone and drone.pos[1] > AIRBORNE_ALT:
                                break
                            time.sleep(CONTROL_DT)

                        final_drone = reader.get_drone()
                        events.log(
                            time.monotonic() - t0,
                            "takeoff_complete",
                            y=float(final_drone.pos[1]) if final_drone else None,
                        )

                        # MPC control loop
                        print(
                            f"MPC teleop active — L to land, Esc to quit  (log: {flight_dir.name}/)"
                        )
                        step = 0
                        prev_mode = "M"
                        prev_loop_start = None
                        active_target = list(TARGET)
                        last_thrust = HOVER_PWM
                        while not stop.is_set():
                            loop_start = time.monotonic()
                            loop_dt_ms = (
                                (loop_start - prev_loop_start) * 1000.0
                                if prev_loop_start is not None
                                else None
                            )
                            prev_loop_start = loop_start

                            if commander.boundary_violated:
                                events.log(
                                    time.monotonic() - t0,
                                    "boundary_violated",
                                    phase="mpc",
                                    pos=list(drone.pos) if drone else None,
                                )
                                print("\n*** BOUNDARY VIOLATED ***")
                                break

                            drone = reader.get_drone()
                            if drone is None:
                                commander.send_setpoint(0.0, 0.0, 0.0, HOVER_PWM)
                                time.sleep(CONTROL_DT)
                                continue

                            mpc = tuner.maybe_rebuild(mpc)
                            x0 = optitrack_to_mpc_state(drone)

                            cone_payload = {"apex": None}

                            target_rb = reader.get_target()
                            if landing_enabled.is_set() and target_rb is not None:
                                ref = landing_reference(
                                    rigid_body_to_state(drone),
                                    rigid_body_to_state(target_rb),
                                    N,
                                    CONTROL_DT,
                                )
                                active_target = [
                                    float(ref[0, 0]),
                                    float(ref[0, 2]),
                                    float(ref[0, 4]),
                                ]
                                TARGET_YAW = target_rb.yaw
                                pad_height = target_rb.pos[1] + 0.05
                                inside_cone = is_in_approach_cone(
                                    drone.pos,
                                    (target_rb.pos[0], pad_height, target_rb.pos[2]),
                                )
                                mode = "L" if inside_cone else "L*"
                                cone_payload = {
                                    "apex": [
                                        float(target_rb.pos[0]),
                                        float(pad_height),
                                        float(target_rb.pos[2]),
                                    ],
                                    "half_angle_deg": float(APPROACH_CONE_HALF_ANGLE_DEG),
                                    "base_radius": float(APPROACH_CONE_BASE_RADIUS_M),
                                    "max_height": float(ARENA_BOUNDS["y_max"]),
                                    "inside": bool(inside_cone),
                                }
                                if drone.pos[1] <= pad_height + TOUCHDOWN_MARGIN and inside_cone:
                                    events.log(
                                        time.monotonic() - t0,
                                        "touchdown",
                                        y=float(drone.pos[1]),
                                        pad_y=float(pad_height),
                                        ramp_duration_s=TOUCHDOWN_RAMP_DURATION,
                                        start_thrust=int(last_thrust),
                                    )
                                    print(
                                        f"\n*** TOUCHDOWN at y={drone.pos[1]:.2f} m — ramping motors down over {TOUCHDOWN_RAMP_DURATION:.1f}s ***"
                                    )
                                    rampdown_start = time.monotonic()
                                    while not stop.is_set():
                                        elapsed = time.monotonic() - rampdown_start
                                        if elapsed >= TOUCHDOWN_RAMP_DURATION:
                                            break
                                        if commander.boundary_violated:
                                            break
                                        frac = 1.0 - elapsed / TOUCHDOWN_RAMP_DURATION
                                        commander.send_setpoint(
                                            0.0, 0.0, 0.0, int(last_thrust * frac)
                                        )
                                        time.sleep(CONTROL_DT)
                                    commander.send_stop_setpoint()
                                    commander.send_notify_setpoint_stop()
                                    events.log(
                                        time.monotonic() - t0,
                                        "motors_off",
                                        y=float(drone.pos[1]) if drone else None,
                                    )
                                    print("*** Motors off ***")
                                    break
                            elif tracking_enabled.is_set() and target_rb is not None:
                                ref = tracking_reference(
                                    rigid_body_to_state(drone),
                                    rigid_body_to_state(target_rb),
                                    N,
                                    CONTROL_DT,
                                )
                                active_target = [
                                    float(ref[0, 0]),
                                    float(ref[0, 2]),
                                    float(ref[0, 4]),
                                ]
                                TARGET_YAW = target_rb.yaw
                                mode = "T"
                            else:
                                if landing_enabled.is_set() and target_rb is None:
                                    landing_enabled.clear()
                                    events.log(
                                        time.monotonic() - t0,
                                        "lost_pose",
                                        prev_mode="L",
                                        topic=TRACKED_OBJECT_NAME,
                                    )
                                    print(
                                        f"\n>>> Landing OFF (lost '{TRACKED_OBJECT_NAME}' pose)"
                                    )
                                if tracking_enabled.is_set() and target_rb is None:
                                    tracking_enabled.clear()
                                    events.log(
                                        time.monotonic() - t0,
                                        "lost_pose",
                                        prev_mode="T",
                                        topic=TRACKED_OBJECT_NAME,
                                    )
                                    print(
                                        f"\n>>> Tracking OFF (lost '{TRACKED_OBJECT_NAME}' pose)"
                                    )
                                if prev_mode in ("T", "L"):
                                    TARGET[:] = active_target
                                update_target(CONTROL_DT)
                                ref = static_reference(TARGET, N, CONTROL_DT)
                                active_target = list(TARGET)
                                mode = "M"

                            if mode != prev_mode:
                                events.log(
                                    time.monotonic() - t0,
                                    "mode_change",
                                    from_mode=prev_mode,
                                    to_mode=mode,
                                )
                            prev_mode = mode

                            u_opt = mpc.compute(x0, ref)
                            ax, ay, az = u_opt

                            yaw = drone.yaw
                            roll, pitch, thrust = mpc_accel_to_cflib_setpoint(
                                ax, ay, az, yaw
                            )
                            yawrate = compute_yawrate(yaw, TARGET_YAW)
                            commander.send_setpoint(roll, pitch, yawrate, thrust)
                            last_thrust = thrust

                            pos = drone.pos
                            vel = drone.vel
                            with battery_lock:
                                vbat = battery_state["vbat"]
                            pos_error = (
                                active_target[0] - pos[0],
                                active_target[1] - pos[1],
                                active_target[2] - pos[2],
                            )
                            vel_error = (
                                float(ref[0, 1]) - vel[0],
                                float(ref[0, 3]) - vel[1],
                                float(ref[0, 5]) - vel[2],
                            )
                            yaw_error = float(wrap_to_pi(TARGET_YAW - yaw))
                            log.log(
                                t=time.monotonic() - t0,
                                pos=pos,
                                vel=vel,
                                target=active_target,
                                accel=u_opt,
                                setpoint=(roll, pitch, thrust),
                                disturbance=mpc.disturbance,
                                yaw=yaw,
                                target_yaw=TARGET_YAW,
                                yawrate=yawrate,
                                config=config,
                                mpc_status=mpc.last_status,
                                mode=mode,
                                solve_time_ms=mpc.last_solve_time_ms,
                                loop_dt_ms=loop_dt_ms,
                                a_cmd_norm=float(np.linalg.norm(u_opt)),
                                vbat=vbat,
                                pos_error=pos_error,
                                vel_error=vel_error,
                                yaw_error=yaw_error,
                                target_vel=(
                                    float(ref[0, 1]),
                                    float(ref[0, 3]),
                                    float(ref[0, 5]),
                                ),
                                pad_pos=tuple(target_rb.pos) if target_rb is not None else None,
                                pad_vel=tuple(target_rb.vel) if target_rb is not None else None,
                            )

                            if mpc.last_status not in ("optimal", "optimal_inaccurate"):
                                infeas.log(
                                    t=time.monotonic() - t0,
                                    x0=x0,
                                    ref=ref,
                                    d_hat=mpc.disturbance,
                                    config=config,
                                    status=mpc.last_status,
                                )

                            step += 1
                            if step % 5 == 0:
                                sys.stdout.write(
                                    f"\r[{mode}] pos=({pos[0]:+.2f},{pos[1]:+.2f},{pos[2]:+.2f}) "
                                    f"err=({pos_error[0]:+.2f},{pos_error[1]:+.2f},{pos_error[2]:+.2f}) "
                                    f"a=({ax:+5.1f},{ay:+5.1f},{az:+5.1f}) "
                                    f"R:{roll:+5.1f} P:{pitch:+5.1f} T:{thrust:5d} "
                                    f"yaw:{np.degrees(yaw):+5.1f}°→{np.degrees(TARGET_YAW):+5.1f}° "
                                    f"yr:{yawrate:+5.1f}°/s "
                                    f"| {tuner.status_line()}  \033[K"
                                )
                                sys.stdout.flush()

                                pub.publish(
                                    MPC_TARGET_TOPIC,
                                    json.dumps(
                                        {"pos": active_target, "yaw": TARGET_YAW}
                                    ),
                                )
                                planned = mpc.get_planned_trajectory()
                                if planned is not None:
                                    pts = [
                                        [float(r[0]), float(r[2]), float(r[4])]
                                        for r in planned
                                    ]
                                    pub.publish(
                                        MPC_TRAJ_TOPIC, json.dumps({"points": pts})
                                    )
                                ref_pts = [
                                    [float(r[0]), float(r[2]), float(r[4])]
                                    for r in ref
                                ]
                                pub.publish(
                                    MPC_REF_TOPIC, json.dumps({"points": ref_pts})
                                )
                                pub.publish(
                                    MPC_CONE_TOPIC, json.dumps(cone_payload)
                                )

                            elapsed = time.monotonic() - loop_start
                            sleep_time = CONTROL_DT - elapsed
                            if sleep_time > 0:
                                time.sleep(sleep_time)

                        # Post-flight: exit session on stop/boundary, else wait for re-takeoff
                        if stop.is_set() or commander.boundary_violated:
                            break
                        mpc._d_hat = 0.0
                        tracking_enabled.clear()
                        landing_enabled.clear()
                        drone = reader.get_drone()
                        if drone is not None:
                            TARGET[:] = [
                                float(drone.pos[0]),
                                HOVER_ALTITUDE,
                                float(drone.pos[2]),
                            ]
                            TARGET_YAW = float(drone.yaw)
                        events.log(
                            time.monotonic() - t0,
                            "ready_for_takeoff",
                            target=list(TARGET),
                        )
                        print(
                            "\n>>> Landed. Press SPACE to take off again (Esc to quit)."
                        )
                        go.clear()
                        go.wait()

            finally:
                commander.send_stop_setpoint()
                commander.send_notify_setpoint_stop()
                time.sleep(0.1)
                battery.stop()
                pub.loop_stop()
                pub.disconnect()
                reader.stop()
                listener.stop()
                print("\nMotors stopped.")


if __name__ == "__main__":
    main()
