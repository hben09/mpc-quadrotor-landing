"""
Minimal MPC hover test — hold position above the origin.

Publishes target to mpc/target so mqtt_viewer shows it.
Press SPACE to take off, Esc to stop.

Usage:
    uv run hover-test
"""

import json
import sys
import time
import threading
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

MQTT_BROKER = "rasticvm.internal"
MQTT_PORT = 1883
DRONE_TOPIC = "rb/crazyflie"
MPC_TARGET_TOPIC = "mpc/target"
MPC_TRAJ_TOPIC = "mpc/trajectory"

RAMP_DURATION = 1.5
AIRBORNE_ALT = 0.3
MIN_POSE_COUNT = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def optitrack_to_mpc_state(rb):
    return np.array([
        rb.pos[0], rb.vel[0],
        rb.pos[1], rb.vel[1],
        rb.pos[2], rb.vel[2],
    ])


def mpc_accel_to_cflib_setpoint(ax, ay, az):
    pitch_deg = float(np.clip(np.degrees(ax / G), -MAX_TILT_DEG, MAX_TILT_DEG))
    roll_deg = float(np.clip(np.degrees(az / G), -MAX_TILT_DEG, MAX_TILT_DEG))
    thrust_pwm = int(np.clip(HOVER_PWM * (1.0 + ay / G), 0, 60000))
    return roll_deg, pitch_deg, 0.0, thrust_pwm


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
            client_id="hover-test-reader",
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
        N = config.horizon

        # MQTT
        reader = DroneStateReader()
        reader.start()

        pub = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id="hover-test-pub",
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

        # Publish target so mqtt_viewer shows it
        pub.publish(MPC_TARGET_TOPIC, json.dumps({"pos": TARGET}))

        # Wait for SPACE
        go = threading.Event()
        stop = threading.Event()

        def on_press(key):
            if key == keyboard.Key.space:
                go.set()
            if key == keyboard.Key.esc:
                stop.set()
                go.set()  # unblock wait

        listener = keyboard.Listener(on_press=on_press)
        listener.start()

        print()
        print(f"=== Hover Test — target {TARGET} ===")
        print("Press SPACE to take off, Esc to abort")
        print("=" * 42)

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

                # MPC hover loop
                print("MPC hover active — Esc to land")
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

                    x0 = optitrack_to_mpc_state(drone)
                    ref = static_reference(TARGET, N, CONTROL_DT)
                    u_opt = mpc.compute(x0, ref)
                    ax, ay, az = u_opt

                    roll, pitch, yawrate, thrust = mpc_accel_to_cflib_setpoint(ax, ay, az)
                    commander.send_setpoint(roll, pitch, yawrate, thrust)

                    step += 1
                    if step % 5 == 0:
                        pos = drone.pos
                        sys.stdout.write(
                            f"\rpos=({pos[0]:+.2f},{pos[1]:+.2f},{pos[2]:+.2f}) "
                            f"R:{roll:+5.1f} P:{pitch:+5.1f} T:{thrust:5d}  "
                        )
                        sys.stdout.flush()

                        pub.publish(MPC_TARGET_TOPIC, json.dumps({"pos": TARGET}))
                        planned = mpc.get_planned_trajectory()
                        if planned is not None:
                            pts = [[float(r[0]), float(r[2]), float(r[4])] for r in planned]
                            pub.publish(MPC_TRAJ_TOPIC, json.dumps({"points": pts}))

                    elapsed = time.monotonic() - loop_start
                    sleep_time = CONTROL_DT - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

            finally:
                commander.send_stop_setpoint()
                commander.send_notify_setpoint_stop()
                time.sleep(0.1)
                pub.loop_stop()
                pub.disconnect()
                reader.stop()
                listener.stop()
                print("\nMotors stopped.")


if __name__ == "__main__":
    main()
