# CLAUDE.md

## Project Overview

MPC-based autonomous quadrotor landing system. A Crazyflie drone is controlled from a PC via Crazyradio 2.0, with state feedback from an OptiTrack motion capture system streamed over MQTT.

## Repository Structure

```
mpc_landing/            # Core control library (workspace member, importable package)
  mpc.py                # Linear MPC controller (CVXPY/OSQP, 3D double integrator)
  reference.py          # Reference trajectory generation (tracking, landing, static)
  boundary.py           # RASTIC arena boundary safety checker
  mqtt/                 # MQTT rigid-body pose streaming from OptiTrack
    parser.py           # Reusable parser: JSON → MQTTRigidBody dataclass with velocity
    sub.py              # MQTT subscriber for drone (crazyflie) and ground vehicle (limo777) poses

sim/                    # Crazyflow simulation environment (workspace member)
  mpc_controller.py     # Closed-loop MPC simulation (hover + tracking + landing)
  mpc_controller_ground.py  # MPC with physics-based ground vehicle
  teleop.py             # Keyboard teleoperation (attitude control, pynput)

hardware/               # Crazyflie hardware control scripts (workspace member)
  keyboard_control.py   # Keyboard teleoperation via cflib (attitude control, 50Hz, pynput)
  thrust_test.py        # Thrust calibration utility for Crazyflie

archive/                # Legacy code kept for reference
  coppeliasim/          # Previous CoppeliaSim-based simulation (replaced by Crazyflow)
```

### Entry Points

All scripts are runnable via `uv run <command>`:
- `keyboard-control` — manual flight with physical Crazyflie
- `thrust-test` — motor thrust calibration
- `sim-mpc` — MPC simulation with virtual target
- `sim-mpc-ground` — MPC simulation with physics-based ground vehicle
- `sim-teleop` — manual flight in Crazyflow simulator

## Architecture

### Control Pipeline
1. **Drone pose** → OptiTrack → Motive → MQTT broker (`rasticvm.internal:1883`) → topic `rb/crazyflie` → `mpc_landing/mqtt/parser.py` → `MQTTRigidBody` (pos, euler, vel)
2. **Ground vehicle pose** → OptiTrack → Motive → MQTT broker → topic `rb/limo777` → `mpc_landing/mqtt/parser.py` → `MQTTRigidBody` (pos, euler, vel)
3. **MPC controller** computes desired acceleration commands
4. **cflib** → Crazyradio 2.0 → Crazyflie

### Current State
- MPC controller implemented in `mpc_landing/mpc.py`, tested in simulation via `sim/mpc_controller.py`
- Hardware control currently via `hardware/keyboard_control.py` (manual attitude teleoperation)
- Next step: integrate MPC with cflib for autonomous hardware flight
- MPC state: [px, vx, py, vy, pz, vz], control: [ax, ay, az], horizon: 25 steps (0.5s)

## Dependencies

- **mpc_landing**: numpy, cvxpy, scipy, paho-mqtt, pyserial
- **sim**: crazyflow (from GitHub), cvxpy, numpy, pynput, scipy
- **hardware**: cflib, pynput

## Hardware

- Drone: Crazyflie 2.1 (Crazyflie firmware), antenna bump indicates the front
- Crazyradio 2.0 USB dongle connected to PC
- OptiTrack motion capture system (3 markers per drone)
- RASTIC arena bounds: X [-4.5, 3.0], Y [0.0, 2.0], Z [-2.0, 3.0] meters
- Hover thrust: ~35000 at full battery

## Key Conventions

- Control loop: 50Hz (20ms period)
- Crazyflie control interface: `cf.commander.send_setpoint(roll, pitch, yawrate, thrust)` — attitude commands via cflib

## Sim vs Hardware Control Interface

MPC (`mpc_landing/mpc.py`) outputs accelerations `[ax, ay, az]`. The translation to actuator commands differs between sim and hardware:

| | Crazyflow sim | Crazyflie hardware (cflib) |
|---|---|---|
| API | `sim.attitude_control([roll, pitch, yaw, thrust])` | `cf.commander.send_setpoint(roll, pitch, yawrate, thrust)` |
| Attitude units | Radians | Degrees |
| Yaw | Absolute angle (rad) | **Rate** (deg/s) |
| Thrust | Newtons | PWM (0–65535) |
| Max tilt | 0.5 rad (~28.6°) | 15° (keyboard_control default) |
| State source | Perfect physics (zero noise/latency) | OptiTrack via MQTT (noisy velocity from finite diff, network latency) |
| Coordinate mapping | `cf_to_mpc_state()` in `sim/mpc_controller.py` | Not yet implemented (needs `optitrack_to_mpc_state()`) |
| Roll/pitch axes | Swapped — see note below | Standard: +pitch = forward, -roll = left |

**Crazyflow roll/pitch visual rotation:** The Crazyflie 3D model in Crazyflow is visually rotated 90° from cflib's convention. The attitude controller itself uses **standard convention** (roll=lateral, pitch=forward in world frame).

- **teleop.py** (human-in-the-loop): Must swap pitch/roll to compensate for the visual model rotation so keyboard controls match what the pilot sees on screen:
  ```
  CF_roll  = radians(cflib_pitch)     # forward/back
  CF_pitch = -radians(cflib_roll)     # left/right
  ```
  See `sim/teleop.py` lines 152-153.
- **MPC** (world-frame control): Uses standard physics (roll=lateral, pitch=forward) and is **not affected** by the visual rotation. `mpc_accel_to_attitude()` in `sim/mpc_controller.py` is correct as-is — no swap needed.

Newton-to-PWM thrust calibration is not yet established; use `hardware/thrust_test.py` to build one.
