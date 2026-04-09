"""
MPC-controlled Crazyflie 2.1 via cflib + OptiTrack/MQTT.

Closed-loop MPC control with mode switching. The drone tracks a ground
vehicle (limo777) whose pose comes from OptiTrack via MQTT.

Modes:
    H  :  HOVER  — hold position at hover target (default on start)
    T  :  TRACK  — follow limo777 at fixed altitude offset
    L  :  LAND   — descend onto limo777 while tracking
    Esc:  emergency stop & quit

Usage:
    uv run mpc-control
"""

import json
import sys
import time
import threading

import numpy as np
from pynput import keyboard

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils import uri_helper

import paho.mqtt.client as mqtt

from mpc_landing import MPCController, MPCConfig
from mpc_landing.boundary import check_boundary
from mpc_landing.mqtt.parser import MQTTRigidBody, RigidBodyTracker
from mpc_landing.reference import landing_reference, static_reference, tracking_reference
from mpc_landing.supervisor import SafeCommander

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
URI = uri_helper.uri_from_env(default="radio://0/80/2M/E7E7E7E7E7")

G = 9.81
CONTROL_HZ = 50
CONTROL_DT = 1.0 / CONTROL_HZ

HOVER_PWM = 45000       # tune via thrust_test.py (battery-dependent)
HOVER_ALTITUDE = 1.0    # metres
MAX_TILT_DEG = 10.0     # conservative start; increase after testing
LANDED_THRESHOLD = 0.10 # metres — cut motors when this close to target
TARGET_SPEED = 0.5      # m/s — WASD manual target speed

MQTT_BROKER = "rasticvm.internal"
MQTT_PORT = 1883
DRONE_TOPIC = "rb/crazyflie"
LIMO_TOPIC = "rb/limo777"
MPC_TARGET_TOPIC = "mpc/target"
MPC_TRAJ_TOPIC = "mpc/trajectory"

# Takeoff ramp
RAMP_DURATION = 1.5     # seconds
AIRBORNE_ALT = 0.3      # metres — hand off to MPC after this

# Startup gating
MIN_POSE_COUNT = 3      # valid poses before engaging MPC


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------
def optitrack_to_mpc_state(rb: MQTTRigidBody) -> np.ndarray:
    """OptiTrack rigid body -> MPC state [px, vx, py, vy, pz, vz].

    OptiTrack and MPC share the same axis order (X=fwd, Y=up, Z=lateral),
    so this is a direct mapping.
    """
    return np.array([
        rb.pos[0], rb.vel[0],  # px, vx (forward)
        rb.pos[1], rb.vel[1],  # py, vy (altitude)
        rb.pos[2], rb.vel[2],  # pz, vz (lateral)
    ])


def mpc_accel_to_cflib_setpoint(ax, ay, az):
    """MPC accelerations -> cflib (roll_deg, pitch_deg, yawrate_deg_s, thrust_pwm).

    Small-angle linearization (same physics as sim/mpc_controller.py):
        pitch = ax / g   (positive ax -> pitch forward)
        roll  = -az / g  (positive az -> need negative roll)

    Thrust: linear model around hover.
        thrust_pwm = HOVER_PWM * (1 + ay / g)
    """
    pitch_deg = float(np.clip(np.degrees(ax / G), -MAX_TILT_DEG, MAX_TILT_DEG))
    roll_deg = float(np.clip(np.degrees(-az / G), -MAX_TILT_DEG, MAX_TILT_DEG))
    yawrate = 0.0
    thrust_pwm = int(np.clip(HOVER_PWM * (1.0 + ay / G), 0, 60000))
    return roll_deg, pitch_deg, yawrate, thrust_pwm


# ---------------------------------------------------------------------------
# MQTT state reader (separate from SafeCommander — needs velocity)
# ---------------------------------------------------------------------------
class MQTTStateReader:
    """Thread-safe MQTT reader for drone and limo state."""

    def __init__(self, broker=MQTT_BROKER, port=MQTT_PORT):
        self._broker = broker
        self._port = port
        self._lock = threading.Lock()

        self._drone_tracker = RigidBodyTracker()
        self._limo_tracker = RigidBodyTracker()
        self._drone: MQTTRigidBody | None = None
        self._limo: MQTTRigidBody | None = None
        self._drone_count = 0

        self._client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id="mpc-state-reader",
            protocol=mqtt.MQTTv311,
        )
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        client.subscribe(DRONE_TOPIC)
        client.subscribe(LIMO_TOPIC)

    def _on_message(self, client, userdata, msg):
        topic = msg.topic
        payload = msg.payload.decode()
        with self._lock:
            if topic == DRONE_TOPIC:
                self._drone = self._drone_tracker.update(payload)
                self._drone_count += 1
            elif topic == LIMO_TOPIC:
                self._limo = self._limo_tracker.update(payload)

    def get_drone(self) -> MQTTRigidBody | None:
        with self._lock:
            return self._drone

    def get_limo(self) -> MQTTRigidBody | None:
        with self._lock:
            return self._limo

    def drone_pose_count(self) -> int:
        with self._lock:
            return self._drone_count

    def start(self):
        self._client.connect(self._broker, self._port)
        self._client.loop_start()

    def stop(self):
        self._client.loop_stop()
        self._client.disconnect()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()


# ---------------------------------------------------------------------------
# Keyboard input
# ---------------------------------------------------------------------------
class KeyState:
    def __init__(self):
        self._lock = threading.Lock()
        self._pressed = set()
        self.exit_flag = False
        self.mode_request = None

    def on_press(self, key):
        if key == keyboard.Key.esc:
            self.exit_flag = True
            return
        try:
            ch = key.char
        except AttributeError:
            return
        with self._lock:
            self._pressed.add(ch)
            if ch == "h":
                self.mode_request = "hover"
            elif ch == "t":
                self.mode_request = "track"
            elif ch == "l":
                self.mode_request = "land"

    def on_release(self, key):
        try:
            ch = key.char
        except AttributeError:
            return
        with self._lock:
            self._pressed.discard(ch)

    def is_pressed(self, char):
        with self._lock:
            return char in self._pressed

    def pop_mode_request(self):
        with self._lock:
            req = self.mode_request
            self.mode_request = None
            return req


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # -- cflib init --
    cflib.crtp.init_drivers()

    print("Scanning for Crazyflie...")
    from pathlib import Path

    available = cflib.crtp.scan_interfaces()
    if not available:
        print("No Crazyflie found! Make sure it is powered on and Crazyradio is plugged in.")
        sys.exit(1)

    uri = available[0][0]
    print(f"Found: {uri}")

    print("Connecting...")
    cache_dir = str(Path(__file__).resolve().parent.parent / "cache")
    with SyncCrazyflie(uri, cf=Crazyflie(rw_cache=cache_dir)) as scf:
        cf = scf.cf

        # Arm and unlock
        cf.platform.send_arming_request(True)
        time.sleep(1.0)
        cf.commander.send_setpoint(0, 0, 0, 0)

        # -- MPC --
        config = MPCConfig(dt=CONTROL_DT, horizon=25)
        mpc = MPCController(config)
        N = config.horizon
        print(f"MPC: horizon={N}, dt={CONTROL_DT}s")

        # -- MQTT state + publisher --
        state_reader = MQTTStateReader()
        state_reader.start()

        mpc_pub = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id="mpc-publisher",
            protocol=mqtt.MQTTv311,
        )
        mpc_pub.connect(MQTT_BROKER, MQTT_PORT)
        mpc_pub.loop_start()

        # Wait for valid drone pose
        print("Waiting for OptiTrack data...", end="", flush=True)
        t0 = time.monotonic()
        while state_reader.drone_pose_count() < MIN_POSE_COUNT:
            if time.monotonic() - t0 > 10.0:
                print("\nNo OptiTrack data received after 10s. Check MQTT broker.")
                state_reader.stop()
                sys.exit(1)
            time.sleep(0.05)
        print(" OK")

        # -- Keyboard --
        keys = KeyState()
        listener = keyboard.Listener(on_press=keys.on_press, on_release=keys.on_release)
        listener.start()

        # -- Modes --
        DISARM, HOVER, TRACK, LAND = "DISARM", "HOVER", "TRACK", "LAND"
        phase = DISARM
        hover_target = [0.0, HOVER_ALTITUDE, 0.0]
        # Manual target (WASD-driven, used as limo substitute)
        target_pos = np.array([0.0, 0.0, 0.0])  # [x, y, z] ground level
        target_vel = np.array([0.0, 0.0, 0.0])

        print()
        print("=== MPC Hardware Control ===")
        print(f"HOVER_PWM={HOVER_PWM}  MAX_TILT={MAX_TILT_DEG}deg  ALT={HOVER_ALTITUDE}m")
        print("H = takeoff & hover | T = track target | L = land")
        print("W/S = target fwd/back | A/D = target left/right | Esc = stop")
        print("=" * 48)
        print("DISARMED — press H to take off")

        with SafeCommander(cf.commander) as commander:
            try:
                # -- Main loop --
                step = 0
                while not keys.exit_flag:
                    loop_start = time.monotonic()

                    if commander.boundary_violated:
                        print("\n*** BOUNDARY VIOLATED ***")
                        break

                    # Mode switching
                    req = keys.pop_mode_request()
                    if req == "hover" and phase == DISARM:
                        # Takeoff ramp
                        print("\nTakeoff ramp...")
                        ramp_start = time.monotonic()
                        while True:
                            if keys.exit_flag or commander.boundary_violated:
                                break
                            elapsed = time.monotonic() - ramp_start
                            if elapsed >= RAMP_DURATION:
                                break
                            frac = elapsed / RAMP_DURATION
                            thrust = int(HOVER_PWM * frac)
                            commander.send_setpoint(0.0, 0.0, 0.0, thrust)

                            drone_rb = state_reader.get_drone()
                            if drone_rb and drone_rb.pos[1] > AIRBORNE_ALT:
                                break  # airborne — hand off to MPC

                            time.sleep(CONTROL_DT)

                        if keys.exit_flag or commander.boundary_violated:
                            break

                        phase = HOVER
                        drone_rb = state_reader.get_drone()
                        if drone_rb:
                            hover_target = [drone_rb.pos[0], HOVER_ALTITUDE, drone_rb.pos[2]]
                        print(f">>> HOVER at {hover_target}")
                    elif req == "hover" and phase != DISARM:
                        phase = HOVER
                        drone_rb = state_reader.get_drone()
                        if drone_rb:
                            hover_target = [drone_rb.pos[0], HOVER_ALTITUDE, drone_rb.pos[2]]
                        print(f"\n>>> HOVER at {hover_target}")
                    elif req == "track" and phase != DISARM:
                        phase = TRACK
                        print("\n>>> TRACK limo")
                    elif req == "land" and phase != DISARM:
                        phase = LAND
                        print("\n>>> LAND on limo")

                    # Update manual target with WASD (always, even while disarmed)
                    vx, vz = 0.0, 0.0
                    if keys.is_pressed("w"):
                        vx += TARGET_SPEED
                    if keys.is_pressed("s"):
                        vx -= TARGET_SPEED
                    if keys.is_pressed("a"):
                        vz -= TARGET_SPEED
                    if keys.is_pressed("d"):
                        vz += TARGET_SPEED
                    target_vel = np.array([vx, 0.0, vz])
                    target_pos += target_vel * CONTROL_DT

                    # Use limo from MQTT if available, otherwise manual target
                    limo_rb = state_reader.get_limo()
                    if limo_rb is not None:
                        target_state = {"pos": limo_rb.pos, "vel": limo_rb.vel}
                        target_alt = limo_rb.pos[1]
                    else:
                        target_state = {
                            "pos": target_pos.tolist(),
                            "vel": target_vel.tolist(),
                        }
                        target_alt = target_pos[1]

                    # Publish target for mqtt_viewer
                    step += 1
                    if step % 5 == 0:
                        mpc_pub.publish(
                            MPC_TARGET_TOPIC,
                            json.dumps({"pos": target_state["pos"]}),
                        )

                    # While disarmed, just idle
                    if phase == DISARM:
                        time.sleep(CONTROL_DT)
                        continue

                    # Read drone state
                    drone_rb = state_reader.get_drone()
                    if drone_rb is None:
                        commander.send_setpoint(0.0, 0.0, 0.0, HOVER_PWM)
                        time.sleep(CONTROL_DT)
                        continue

                    x0 = optitrack_to_mpc_state(drone_rb)
                    drone_state = {
                        "pos": [x0[0], x0[2], x0[4]],
                        "vel": [x0[1], x0[3], x0[5]],
                    }

                    # Build reference
                    if phase == HOVER:
                        hover_target[0] = target_state["pos"][0]
                        hover_target[2] = target_state["pos"][2]
                        ref = static_reference(hover_target, N, CONTROL_DT)
                    elif phase == TRACK:
                        ref = tracking_reference(drone_state, target_state, N, CONTROL_DT)
                    elif phase == LAND:
                        ref = landing_reference(drone_state, target_state, N, CONTROL_DT)
                        if drone_rb.pos[1] - target_alt < LANDED_THRESHOLD:
                            print("\n>>> LANDED — cutting motors")
                            break

                    # Solve MPC
                    u_opt = mpc.compute(x0, ref)
                    ax, ay, az = u_opt

                    # MPC solver failure — hold hover
                    if np.allclose(u_opt, 0.0) and phase != HOVER:
                        roll, pitch, yawrate, thrust = 0.0, 0.0, 0.0, HOVER_PWM
                    else:
                        roll, pitch, yawrate, thrust = mpc_accel_to_cflib_setpoint(ax, ay, az)

                    commander.send_setpoint(roll, pitch, yawrate, thrust)

                    # Log + publish at 10 Hz
                    if step % 5 == 0:
                        pos = drone_rb.pos
                        tp = target_state["pos"]
                        sys.stdout.write(
                            f"\r[{phase:>5s}] "
                            f"pos=({pos[0]:+.2f},{pos[1]:+.2f},{pos[2]:+.2f}) "
                            f"tgt=({tp[0]:+.2f},{tp[1]:+.2f},{tp[2]:+.2f}) "
                            f"R:{roll:+5.1f} P:{pitch:+5.1f} T:{thrust:5d}  "
                        )
                        sys.stdout.flush()

                        # Publish trajectory for mqtt_viewer
                        planned = mpc.get_planned_trajectory()
                        if planned is not None:
                            pts = [[float(r[0]), float(r[2]), float(r[4])] for r in planned]
                            mpc_pub.publish(MPC_TRAJ_TOPIC, json.dumps({"points": pts}))

                    # Maintain 50 Hz
                    elapsed = time.monotonic() - loop_start
                    sleep_time = CONTROL_DT - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

            finally:
                commander.send_stop_setpoint()
                commander.send_notify_setpoint_stop()
                time.sleep(0.1)
                mpc_pub.loop_stop()
                mpc_pub.disconnect()
                state_reader.stop()
                listener.stop()
                print("\nDisconnected. Motors stopped.")


if __name__ == "__main__":
    main()
