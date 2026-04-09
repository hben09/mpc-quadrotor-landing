"""
MPC-controlled quadrotor in CoppeliaSim.

Modified to track a real ground bot object in the scene instead of a virtual
keyboard-driven target. The ground bot pose is read from CoppeliaSim every
control step and converted into the MPC reference.

Modes:
    1. HOVER  — drone takes off and holds position above origin
    2. TRACK  — press T to start tracking the ground bot with altitude offset
    3. LAND   — press L to descend onto the ground bot position

Keyboard:
    R    : arm/ enable mpc outputs
    F    : disarm/ cut throttle
    T    : switch to TRACK mode
    L    : switch to LAND mode
    H    : return to HOVER at origin
    Esc  : stop simulation

Coordinate mapping (CoppeliaSim z-up -> MPC convention):
    sim x  ->  MPC px  (horizontal 1)
    sim z  ->  MPC py  (altitude / vertical)
    sim y  ->  MPC pz  (horizontal 2)
"""

import csv
import json
import sys
import time
import threading
from datetime import datetime
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
# CoppeliaSim object names
# ---------------------------------------------------------------------------
DRONE_NAME = "/Quadcopter"
GROUND_NAME = "/Ground"

# ---------------------------------------------------------------------------
# Control parameters
# ---------------------------------------------------------------------------
CONTROL_HZ = 50
CONTROL_DT = 1.0 / CONTROL_HZ

# PWM constants (must match Lua flight controller)
PWM_CENTER = 1500
PWM_MIN = 1000
PWM_MAX = 2000
HOVER_PWM = 1502  # empirically determined hover bias

# Acceleration -> PWM gains (tuned empirically)
K_PITCH = 40.0     # PWM/(m/s^2)
K_ROLL = 8.0
K_THRUST = 10.0   # PWM/(m/s^2)
MAX_TILT_PWM = 60  # max roll/pitch PWM offset from center (tuned empirically)

# Tracking parameters
HOVER_ALTITUDE = 0.5
TRACKING_ALTITUDE = 0.2    # meters above ground bot in tracking mode
GROUND_V_DEFAULT = 0.4     # default forward velocity signal for ground bot
SHOW_TARGET_MARKER = True
DATA_DIR = Path(__file__).resolve().parent / "data"


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
    """MPC accelerations -> PWM using a small-angle linearization."""
    pitch_pwm = PWM_CENTER + K_PITCH * ax
    thrust_pwm = HOVER_PWM + K_THRUST * ay
    roll_pwm = PWM_CENTER - K_ROLL * az   # inverted sign for sim y-axis response
    yaw_pwm = PWM_CENTER

    return (
        clamp(roll_pwm, PWM_CENTER - MAX_TILT_PWM, PWM_CENTER + MAX_TILT_PWM),
        clamp(pitch_pwm, PWM_CENTER - MAX_TILT_PWM, PWM_CENTER + MAX_TILT_PWM),
        clamp(yaw_pwm, PWM_MIN, PWM_MAX),
        clamp(thrust_pwm, PWM_MIN, PWM_MAX),
    )


def send_commands(sim, roll, pitch, yaw, thrust):
    sim.setFloatSignal("cmd_roll", float(int(round(roll))))
    sim.setFloatSignal("cmd_pitch", float(int(round(pitch))))
    sim.setFloatSignal("cmd_yaw", float(int(round(yaw))))
    sim.setFloatSignal("cmd_thrust", float(int(round(thrust))))


def read_ground_state(sim, ground_handle):
    """
    Read the real ground bot state and convert it into the state dictionary
    expected by reference.py.

    Returned coordinates use MPC convention:
        pos = [px, py, pz] = [sim x, sim z, sim y]
    """
    pos = sim.getObjectPosition(ground_handle, sim.handle_world)

    return {
        "pos": [pos[0], pos[2], pos[1]],
        "sim_pos": pos,
    }


def _serialize_value(value):
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value)
    return value


def write_simulation_parameters(params_path, config):
    """Write the fixed parameters used for this run."""
    parameters = {
        "script": Path(__file__).name,
        "drone_name": DRONE_NAME,
        "ground_name": GROUND_NAME,
        "control_hz": CONTROL_HZ,
        "control_dt": CONTROL_DT,
        "pwm_center": PWM_CENTER,
        "pwm_min": PWM_MIN,
        "pwm_max": PWM_MAX,
        "hover_pwm": HOVER_PWM,
        "k_pitch": K_PITCH,
        "k_roll": K_ROLL,
        "k_thrust": K_THRUST,
        "max_tilt_pwm": MAX_TILT_PWM,
        "hover_altitude": HOVER_ALTITUDE,
        "tracking_altitude": TRACKING_ALTITUDE,
        "ground_v_default": GROUND_V_DEFAULT,
        "show_target_marker": SHOW_TARGET_MARKER,
        "mpc_dt": config.dt,
        "mpc_horizon": config.horizon,
        "mpc_q_diag": config.Q_diag,
        "mpc_qf_diag": config.Qf_diag,
        "mpc_r_diag": config.R_diag,
        "mpc_a_max": config.a_max,
        "mpc_v_max": config.v_max,
    }

    with params_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["parameter", "value"])
        for name, value in parameters.items():
            writer.writerow([name, _serialize_value(value)])


def create_run_logs(config):
    """Create CSV files for the run trace and simulation parameters."""
    DATA_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = DATA_DIR / f"run_{timestamp}"
    run_dir.mkdir()
    trace_path = run_dir / "simulation_trace.csv"
    params_path = run_dir / "simulation_params.csv"

    trace_file = trace_path.open("w", newline="")
    trace_writer = csv.writer(trace_file)
    trace_writer.writerow(
        [
            "time_s",
            "phase",
            "armed",
            "key_r",
            "key_f",
            "key_t",
            "key_l",
            "key_h",
            "drone_x",
            "drone_y",
            "drone_z",
            "drone_vx",
            "drone_vy",
            "drone_vz",
            "ground_x",
            "ground_y",
            "ground_z",
            "ground_vx",
            "ground_vy",
            "ground_vz",
            "target_x",
            "target_y",
            "target_z",
            "ax_cmd",
            "ay_cmd",
            "az_cmd",
            "roll_pwm",
            "pitch_pwm",
            "yaw_pwm",
            "thrust_pwm",
        ]
    )

    write_simulation_parameters(params_path, config)
    return run_dir, trace_path, params_path, trace_file, trace_writer


# ---------------------------------------------------------------------------
# Keyboard input
# ---------------------------------------------------------------------------
class KeyState:
    def __init__(self):
        self._pressed = set()
        self._lock = threading.Lock()
        self.exit_flag = False
        self.mode_request = None  # "track", "land", "hover"
        self.arm_request = None   # "arm", "disarm"

    def pop_arm_request(self):
        req = self.arm_request
        self.arm_request = None
        return req

    def on_press(self, key):
        try:
            k = key.char.lower()
        except AttributeError:
            k = key
        with self._lock:
            self._pressed.add(k)

        if hasattr(key, "char"):
            if k == "t":
                self.mode_request = "track"
            elif k == "l":
                self.mode_request = "land"
            elif k == "h":
                self.mode_request = "hover"
            elif k == "r":
                self.arm_request = "arm"
            elif k == "f":
                self.arm_request = "disarm"

        if key == keyboard.Key.esc:
            self.exit_flag = True

    def on_release(self, key):
        try:
            k = key.char.lower()
        except AttributeError:
            k = key
        with self._lock:
            self._pressed.discard(k)

    def pop_mode_request(self):
        req = self.mode_request
        self.mode_request = None
        return req

    def pressed_snapshot(self):
        with self._lock:
            return set(self._pressed)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Connecting to CoppeliaSim...")
    client = RemoteAPIClient()
    sim = client.getObject("sim")

    drone_handle = sim.getObject(DRONE_NAME)
    ground_handle = sim.getObject(GROUND_NAME)
    print(f"Found drone:  {DRONE_NAME}")
    print(f"Found ground: {GROUND_NAME}")

    # MPC
    config = MPCConfig(dt=CONTROL_DT, horizon=40)
    mpc = MPCController(config)
    N = config.horizon
    print(f"MPC: horizon={N}, dt={CONTROL_DT}s")

    # Optional marker that shows the current tracking target
    target_marker = -1
    if SHOW_TARGET_MARKER:
        target_marker = sim.createPrimitiveShape(2, [0.1, 0.1, 0.1])  # sphere
        sim.setObjectAlias(target_marker, "Target")
        sim.setShapeColor(target_marker, None, 0, [1.0, 0.2, 0.2])
        sim.setObjectInt32Param(target_marker, sim.shapeintparam_static, 1)
        sim.setObjectInt32Param(target_marker, sim.shapeintparam_respondable, 0)

    # Keyboard
    keys = KeyState()
    listener = keyboard.Listener(on_press=keys.on_press, on_release=keys.on_release)
    listener.start()

    # Mode state
    HOVER = "hover"
    TRACK = "track"
    LAND = "land"
    armed = False
    phase = HOVER
    hover_target = None
    track_altitude = TRACKING_ALTITUDE
    run_dir, trace_path, params_path, trace_file, trace_writer = create_run_logs(config)
    sim_start_time = None


    sim.startSimulation()
    time.sleep(0.2)
    sim_start_time = time.time()
    drone_pos0 = sim.getObjectPosition(drone_handle, sim.handle_world)
    hover_target = [drone_pos0[0], HOVER_ALTITUDE, drone_pos0[1]]
    print(f"Saving run data to: {run_dir}")
    print(f"Saving simulation trace to: {trace_path}")
    print(f"Saving simulation parameters to: {params_path}")

    print("\n=== MPC Quadrotor Ground-Tracking Controller ===")
    print(f"Drone rises to {HOVER_ALTITUDE} m and holds position on start.")
    print("Then it can track or land on the real /Ground robot.")
    print()
    print("Controls:")
    print("  T   : switch to TRACK mode")
    print("  L   : switch to LAND mode")
    print("  Esc : quit")
    print("=" * 48)

    step_count = 0
    try:
        while sim.getSimulationState() != sim.simulation_stopped and not keys.exit_flag:
            # Keep the ground bot moving with its Lua signal interface
            sim.setFloatSignal("ground_v", GROUND_V_DEFAULT)

            # --- Mode switches ---
            req = keys.pop_mode_request()
            if req == "track" and phase != TRACK:
                phase = TRACK
                drone_pos_now = sim.getObjectPosition(drone_handle, sim.handle_world)
                track_altitude = TRACKING_ALTITUDE   # fixed tracking altitude
                print(f"\n>>> Mode: TRACK — following /Ground at altitude {track_altitude} relative to limobot")
            elif req == "land" and phase != LAND:
                phase = LAND
                print("\n>>> Mode: LAND — descending onto /Ground")
            elif req == "hover":
                phase = HOVER
                drone_pos_now = sim.getObjectPosition(drone_handle, sim.handle_world)
                hover_target = [drone_pos_now[0], drone_pos_now[2], drone_pos_now[1]]
                print("\n>>> Mode: HOVER — holding current position")
            
            # --- Arm / disarm ---
            arm_req = keys.pop_arm_request()
            if arm_req == "arm" and not armed:
                armed = True
                print("\n>>> Controller ARMED — MPC outputs enabled")
            elif arm_req == "disarm" and armed:
                armed = False
                send_commands(sim, PWM_CENTER, PWM_CENTER, PWM_CENTER, PWM_MIN)
                print("\n>>> Controller DISARMED — throttle cut, neutral attitude")
                
            # --- Read drone state ---
            drone_pos = sim.getObjectPosition(drone_handle, sim.handle_world)
            drone_vel, _ = sim.getObjectVelocity(drone_handle)
            x0 = sim_to_mpc_state(drone_pos, drone_vel)

            # --- Read real ground bot state ---
            ground_state = read_ground_state(sim, ground_handle)

            # Raise the tracking/landing reference to stay above the robot
            target_pos = ground_state["pos"].copy()
            if phase == TRACK:
                target_pos[1] = ground_state["pos"][1] + track_altitude
            elif phase == LAND:
                target_pos[1] += 0.0

            ground_vel = sim.getObjectVelocity(ground_handle)[0]

            ref_target_state = {
                "pos": target_pos,
                "vel": [ground_vel[0], ground_vel[2], ground_vel[1]],
            }

            # --- Build reference ---
            if phase == HOVER:
                ref = static_reference(hover_target, N, CONTROL_DT)
            else:
                drone_state = {
                    "pos": [x0[0], x0[2], x0[4]],
                    "vel": [x0[1], x0[3], x0[5]],
                }
                if phase == TRACK:
                    ref = tracking_reference(drone_state, ref_target_state, N, CONTROL_DT)
                else:
                    ref = landing_reference(drone_state, ref_target_state, N, CONTROL_DT)

            # --- Move target marker (MPC convention -> sim x=px, y=pz, z=py) ---
            if target_marker != -1:
                sim.setObjectPosition(
                    target_marker,
                    sim.handle_world,
                    [target_pos[0], target_pos[2], target_pos[1]],
                )

            # --- Solve MPC ---
            u_opt = mpc.compute(x0, ref)
            ax, ay, az = u_opt

            # --- Send PWM only if armed ---
            roll_pwm, pitch_pwm, yaw_pwm, thrust_pwm = mpc_accel_to_pwm(ax, ay, az)

            if armed:
                send_commands(sim, roll_pwm, pitch_pwm, yaw_pwm, thrust_pwm)
            else:
                roll_pwm = PWM_CENTER
                pitch_pwm = PWM_CENTER
                yaw_pwm = PWM_CENTER
                thrust_pwm = PWM_MIN
                send_commands(sim, roll_pwm, pitch_pwm, yaw_pwm, thrust_pwm)

            pressed_keys = keys.pressed_snapshot()
            trace_writer.writerow(
                [
                    time.time() - sim_start_time,
                    phase,
                    int(armed),
                    int("r" in pressed_keys),
                    int("f" in pressed_keys),
                    int("t" in pressed_keys),
                    int("l" in pressed_keys),
                    int("h" in pressed_keys),
                    drone_pos[0],
                    drone_pos[1],
                    drone_pos[2],
                    drone_vel[0],
                    drone_vel[1],
                    drone_vel[2],
                    ground_state["sim_pos"][0],
                    ground_state["sim_pos"][1],
                    ground_state["sim_pos"][2],
                    ground_vel[0],
                    ground_vel[1],
                    ground_vel[2],
                    target_pos[0],
                    target_pos[2],
                    target_pos[1],
                    ax,
                    ay,
                    az,
                    roll_pwm,
                    pitch_pwm,
                    yaw_pwm,
                    thrust_pwm,
                ]
            )
            trace_file.flush()

            # --- Logging ---
            step_count += 1
            if step_count % 5 == 0:
                gp = ground_state["sim_pos"]
                print(
                    f"[{phase:>5s}] "
                    f"drone=({drone_pos[0]:+.2f},{drone_pos[1]:+.2f},{drone_pos[2]:+.2f})  "
                    f"ground=({gp[0]:+.2f},{gp[1]:+.2f},{gp[2]:+.2f})  "
                    f"target_mpc=({target_pos[0]:+.2f},{target_pos[1]:+.2f},{target_pos[2]:+.2f})  "
                    f"accel=({ax:+.2f},{ay:+.2f},{az:+.2f})  "
                    f"PWM: R={roll_pwm:.0f} P={pitch_pwm:.0f} T={thrust_pwm:.0f}"
                )

            time.sleep(CONTROL_DT)

    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        listener.stop()
        trace_file.close()
        send_commands(sim, PWM_CENTER, PWM_CENTER, PWM_CENTER, PWM_MIN)
        time.sleep(0.1)
        if target_marker != -1:
            sim.removeObject(target_marker)
        sim.stopSimulation()
        print(f"Run data saved to: {run_dir}")
        print(f"Simulation trace saved to: {trace_path}")
        print(f"Simulation parameters saved to: {params_path}")
        print("\nSimulation stopped.")


if __name__ == "__main__":
    main()
