"""
MPC teleop — fly via a manually-piloted setpoint.

Publishes target to mpc/target so dashboard shows it.
Press SPACE to take off, Esc to stop.

Usage:
    uv run mpc-teleop
"""

import csv
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
from mpc_landing.mqtt.parser import RigidBodyTracker
from mpc_landing.reference import static_reference
from mpc_landing.supervisor import SafeCommander
from mpc_landing.yaw_controller import compute_yawrate, wrap_to_pi

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
MPC_TARGET_TOPIC = "mpc/target"
MPC_TRAJ_TOPIC = "mpc/trajectory"

TARGET_SPEED = 0.5  # meters per second (WASD/QE, continuous while held)

RAMP_DURATION = 1.5
AIRBORNE_ALT = 0.3
MIN_POSE_COUNT = 3

# Pressed-key tracking for smooth target movement
pressed_keys = set()
keys_lock = threading.Lock()


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
# MQTT state reader (drone only)
# ---------------------------------------------------------------------------
class DroneStateReader:
    def __init__(self):
        self._lock = threading.Lock()
        self._tracker = RigidBodyTracker()
        self._drone = None
        self._count = 0
        self._client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id="mpc-teleop-reader",
            protocol=mqtt.MQTTv311,
        )
        self._client.on_connect = lambda c, u, f, rc, p: c.subscribe(DRONE_TOPIC)
        self._client.on_message = self._on_msg

    def _on_msg(self, client, userdata, msg):
        with self._lock:
            self._drone = self._tracker.update(msg.payload.decode())
            self._count += 1

    def get(self):
        with self._lock:
            return self._drone

    def count(self):
        with self._lock:
            return self._count

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
        config = MPCConfig(dt=CONTROL_DT, horizon=25)
        mpc = MPCController(config)
        tuner = ParamTuner(config)
        N = config.horizon

        # MQTT
        reader = DroneStateReader()
        reader.start()

        pub = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id="mpc-teleop-pub",
            protocol=mqtt.MQTTv311,
        )
        pub.connect(MQTT_BROKER, MQTT_PORT)
        pub.loop_start()

        # Wait for OptiTrack
        print("Waiting for OptiTrack...", end="", flush=True)
        t0 = time.monotonic()
        while reader.count() < MIN_POSE_COUNT:
            if time.monotonic() - t0 > 10.0:
                print(" TIMEOUT")
                reader.stop()
                pub.loop_stop()
                sys.exit(1)
            time.sleep(0.05)
        print(" OK")

        # Publish target so dashboard shows it
        pub.publish(MPC_TARGET_TOPIC, json.dumps({"pos": TARGET, "yaw": TARGET_YAW}))

        # Wait for SPACE
        go = threading.Event()
        stop = threading.Event()

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
            # Track held keys for smooth target movement
            with keys_lock:
                try:
                    if hasattr(key, "char") and key.char:
                        c = key.char.lower()
                        if c in "123456":
                            tuner.select(int(c) - 1)
                        pressed_keys.add(c)
                except AttributeError:
                    pressed_keys.add(key)

        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()

        print()
        print(f"=== MPC Teleop — target {TARGET} ===")
        print("Press SPACE to take off, Esc to abort")
        print("WASD = move XZ, Z/X = altitude, Q/E = yaw target (hold)")
        print("MPC tuning: 1-6 select, Up/Down adjust")
        print("  1:Q_pos 2:Q_vel 3:Qf 4:R 5:a_max 6:v_max")
        print("=" * 46)

        go.wait()
        if stop.is_set():
            print("Aborted.")
            reader.stop()
            pub.loop_stop()
            pub.disconnect()
            listener.stop()
            return

        with SafeCommander(cf.commander) as commander:
            try:
                # Takeoff ramp
                print("Takeoff ramp...")
                ramp_start = time.monotonic()
                while not stop.is_set():
                    elapsed = time.monotonic() - ramp_start
                    if elapsed >= RAMP_DURATION:
                        break
                    if commander.boundary_violated:
                        break
                    frac = elapsed / RAMP_DURATION
                    commander.send_setpoint(0.0, 0.0, 0.0, int(HOVER_PWM * frac))
                    drone = reader.get()
                    if drone and drone.pos[1] > AIRBORNE_ALT:
                        break
                    time.sleep(CONTROL_DT)

                # MPC control loop
                log_dir = Path(__file__).resolve().parent / "logs"
                log_dir.mkdir(exist_ok=True)
                log_path = log_dir / f"teleop_{datetime.now():%Y%m%d_%H%M%S}.csv"
                log_file = open(log_path, "w", newline="")
                log_writer = csv.writer(log_file)
                log_writer.writerow(
                    [
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
                )
                t0_mpc = time.monotonic()
                print(f"MPC teleop active — Esc to land  (log: {log_path.name})")
                step = 0
                while not stop.is_set():
                    loop_start = time.monotonic()

                    if commander.boundary_violated:
                        print("\n*** BOUNDARY VIOLATED ***")
                        break

                    drone = reader.get()
                    if drone is None:
                        commander.send_setpoint(0.0, 0.0, 0.0, HOVER_PWM)
                        time.sleep(CONTROL_DT)
                        continue

                    mpc = tuner.maybe_rebuild(mpc)
                    update_target(CONTROL_DT)
                    x0 = optitrack_to_mpc_state(drone)
                    ref = static_reference(TARGET, N, CONTROL_DT)
                    u_opt = mpc.compute(x0, ref)
                    ax, ay, az = u_opt

                    yaw = drone.yaw
                    roll, pitch, thrust = mpc_accel_to_cflib_setpoint(ax, ay, az, yaw)
                    yawrate = compute_yawrate(yaw, TARGET_YAW)
                    commander.send_setpoint(roll, pitch, yawrate, thrust)

                    # Log every step to CSV
                    pos = drone.pos
                    vel = drone.vel
                    log_writer.writerow(
                        [
                            f"{time.monotonic() - t0_mpc:.3f}",
                            f"{pos[0]:.4f}",
                            f"{pos[1]:.4f}",
                            f"{pos[2]:.4f}",
                            f"{vel[0]:.4f}",
                            f"{vel[1]:.4f}",
                            f"{vel[2]:.4f}",
                            f"{TARGET[0]:.4f}",
                            f"{TARGET[1]:.4f}",
                            f"{TARGET[2]:.4f}",
                            f"{ax:.4f}",
                            f"{ay:.4f}",
                            f"{az:.4f}",
                            f"{roll:.2f}",
                            f"{pitch:.2f}",
                            f"{thrust}",
                            f"{mpc.disturbance:.4f}",
                            f"{yaw:.4f}",
                            f"{TARGET_YAW:.4f}",
                            f"{yawrate:.2f}",
                            f"{config.Q_diag[0]:.2f}",
                            f"{config.Q_diag[1]:.2f}",
                            f"{config.Qf_diag[0]:.2f}",
                            f"{config.R_diag[0]:.2f}",
                            f"{config.a_max:.2f}",
                            f"{config.v_max:.2f}",
                        ]
                    )

                    step += 1
                    if step % 5 == 0:
                        err = [TARGET[i] - pos[i] for i in range(3)]
                        sys.stdout.write(
                            f"\rpos=({pos[0]:+.2f},{pos[1]:+.2f},{pos[2]:+.2f}) "
                            f"err=({err[0]:+.2f},{err[1]:+.2f},{err[2]:+.2f}) "
                            f"a=({ax:+5.1f},{ay:+5.1f},{az:+5.1f}) "
                            f"R:{roll:+5.1f} P:{pitch:+5.1f} T:{thrust:5d} "
                            f"yaw:{np.degrees(yaw):+5.1f}°→{np.degrees(TARGET_YAW):+5.1f}° "
                            f"yr:{yawrate:+5.1f}°/s "
                            f"| {tuner.status_line()}  \033[K"
                        )
                        sys.stdout.flush()

                        pub.publish(
                            MPC_TARGET_TOPIC,
                            json.dumps({"pos": TARGET, "yaw": TARGET_YAW}),
                        )
                        planned = mpc.get_planned_trajectory()
                        if planned is not None:
                            pts = [
                                [float(r[0]), float(r[2]), float(r[4])] for r in planned
                            ]
                            pub.publish(MPC_TRAJ_TOPIC, json.dumps({"points": pts}))

                    elapsed = time.monotonic() - loop_start
                    sleep_time = CONTROL_DT - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

            finally:
                commander.send_stop_setpoint()
                commander.send_notify_setpoint_stop()
                log_file.close()
                time.sleep(0.1)
                pub.loop_stop()
                pub.disconnect()
                reader.stop()
                listener.stop()
                print("\nMotors stopped.")


if __name__ == "__main__":
    main()
