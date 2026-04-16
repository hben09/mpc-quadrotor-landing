# CLAUDE.md

## Project Overview

MPC-based autonomous quadrotor landing system. A Crazyflie drone is controlled from a PC via Crazyradio 2.0, with state feedback from an OptiTrack motion capture system streamed over MQTT.

## Repository Structure

```
mpc_landing/            # Core control library (workspace member, importable package)
  mpc.py                # Offset-free MPC controller (CVXPY/OSQP, 3D double integrator + vertical disturbance state)
  reference.py          # Reference trajectory generation (tracking, landing, static)
  boundary.py           # RASTIC arena boundary safety checker
  supervisor.py         # SafeCommander — boundary safety supervisor wrapping cf.commander
  yaw_controller.py     # P yaw controller (OptiTrack CCW+ → cflib CW+ sign flip, deg/s output)
  mqtt/                 # MQTT rigid-body pose streaming from OptiTrack
    parser.py           # Reusable parser: JSON → MQTTRigidBody dataclass with velocity
    sub.py              # MQTT subscriber for drone (crazyflie) and landing target (rb/landing) poses

sim/                    # Crazyflow simulation environment (workspace member)
  mpc_controller.py     # Closed-loop MPC simulation (hover + tracking + landing)
  mpc_controller_ground.py  # MPC with physics-based ground vehicle
  mpc_gui.py            # PyVista GUI — drag target sphere, shows MPC plan + arena bounds
  simu_mpc_teleop_landing.py  # Sim MPC tracking + autonomous descent on rb/landing
  teleop.py             # Keyboard teleoperation (attitude control, pynput)

hardware/               # Crazyflie hardware control scripts (workspace member)
  teleop.py             # Keyboard teleoperation via cflib (attitude control, 50Hz, pynput) — keep barebones
  mpc_teleop.py         # MPC position controller with manually-piloted setpoint (WASD/QE, runtime tuning, CSV logging)
  mpc_teleop_landing.py # MPC tracking + autonomous descent on rb/landing (physical Crazyflie)
  battery.py            # BatteryPublisher — cflib pm.* LogConfig → MQTT topic cf/battery
  dashboard.py          # Real-time 3D drone position viewer via MQTT/OptiTrack (PyVista)
```

### Entry Points

All scripts are runnable via `uv run <command>`:
- `hw-teleop` — manual attitude flight with physical Crazyflie
- `mpc-teleop` — MPC position flight with manually-piloted setpoint (physical Crazyflie)
- `dashboard` — real-time 3D drone position viewer (OptiTrack via MQTT)
- `sim-mpc` — MPC simulation with virtual target
- `sim-mpc-ground` — MPC simulation with physics-based ground vehicle
- `sim-mpc-gui` — interactive PyVista GUI for sim MPC (drag target sphere)
- `sim-teleop` — manual flight in Crazyflow simulator

The autonomous landing scripts have no console-script alias yet — invoke them directly:
- `uv run python hardware/mpc_teleop_landing.py` — MPC tracking + autonomous descent on `rb/landing` (physical Crazyflie)
- `uv run python sim/simu_mpc_teleop_landing.py` — same flow, in Crazyflow sim

## Architecture

### Control Pipeline
1. **Drone pose** → OptiTrack → Motive → MQTT broker (`rasticvm.internal:1883`) → topic `rb/crazyflie` → `mpc_landing/mqtt/parser.py` → `MQTTRigidBody` (pos, euler, vel)
2. **Landing target pose** → OptiTrack → Motive → MQTT broker → topic `rb/landing` → `mpc_landing/mqtt/parser.py` → `MQTTRigidBody` (pos, euler, vel)
3. **MPC controller** computes desired acceleration commands
4. **cflib** → Crazyradio 2.0 → Crazyflie

### Boundary Supervisor (`SafeCommander`)
`mpc_landing/supervisor.py` provides `SafeCommander`, a drop-in replacement for `cf.commander` that enforces arena boundary safety. It:
- Monitors drone position via MQTT (`rb/crazyflie`) in a background thread
- Checks every `send_setpoint()` call against arena boundaries (via `check_boundary()`)
- Disarms motors and blocks all further commands if:
  - Drone violates a boundary (only checked once airborne, Y > 0.3 m)
  - Position data goes stale (>2 s without MQTT update)
- Degrades gracefully: if the MQTT broker is unreachable, prints a warning and passes commands through unsupervised

Usage (context manager):
```python
with SafeCommander(cf.commander) as commander:
    commander.send_setpoint(roll, pitch, yawrate, thrust)
```
Currently used in `hardware/teleop.py`.

### Current State
- MPC controller implemented in `mpc_landing/mpc.py`, tested in simulation via `sim/mpc_controller.py`
- Hardware MPC position flight via `hardware/mpc_teleop.py` (manually-piloted setpoint, yaw-compensated accel→attitude mapping)
- Autonomous tracking + landing on `rb/landing` via `hardware/mpc_teleop_landing.py` (hardware) and `sim/simu_mpc_teleop_landing.py` (sim), using `tracking_reference()` + `landing_reference()` from `mpc_landing/reference.py`
- Yaw P-controller in `mpc_landing/yaw_controller.py` runs alongside position MPC (decoupled world-frame axis)
- Manual attitude teleoperation still available via `hardware/teleop.py`
- MPC state: [px, vx, py, vy, pz, vz, d] (7D — d is vertical disturbance for offset-free tracking), control: [ax, ay, az], horizon: 25 steps (0.5s)

## Dependencies

- **mpc_landing**: numpy, cvxpy, scipy, paho-mqtt, pyserial
- **sim**: crazyflow (from GitHub), cvxpy, numpy, pynput, scipy
- **hardware**: cflib, pynput

## Hardware

- Drone: Crazyflie 2.1 (Crazyflie firmware), antenna bump indicates the front
- Crazyradio 2.0 USB dongle connected to PC
- OptiTrack motion capture system (4 markers per drone)
- OptiTrack/Motive coordinate system: X = forward, Y = up, Z = right
- RASTIC arena bounds: X [-4.5, 3.0], Y [0.0, 2.0], Z [-2.0, 3.0] meters
- Hover thrust: ~45000 at full battery (with 3 markers)

## Key Conventions

- Control loop: 50Hz (20ms period)
- Crazyflie control interface: `cf.commander.send_setpoint(roll, pitch, yawrate, thrust)` — attitude commands via cflib
- **`hardware/teleop.py` stays barebones** — manual attitude flight only. Do NOT wire MPC, MQTT publishing, battery logging, or other telemetry into it. It exists as the minimal "just fly the drone" fallback. Put such features in `mpc_teleop.py` / `mpc_teleop_landing.py` instead.

## Coordinate Systems

| | Forward | Up | Lateral |
|---|---|---|---|
| **OptiTrack/Motive** | X | Y | Z |
| **Crazyflow sim** | X | Z | Y |
| **MPC state** [px, py, pz] | px | py | pz |

OptiTrack and MPC use the same axis order (no swap needed). Crazyflow swaps Y/Z — see `cf_to_mpc_state()` in `sim/mpc_controller.py`.

### OptiTrack Euler Angles
`MQTTRigidBody.euler` is computed via `Rotation.as_euler("xyz")` (intrinsic XYZ). In OptiTrack's frame the indices map to:
- `euler[0]` — **Roll** (rotation about X/forward): + = right roll
- `euler[1]` — **Yaw** (rotation about Y/up): + = left (CCW from above)
- `euler[2]` — **Pitch** (rotation about Z/right): + = nose up (backward pitch)

⚠ **`euler[1]` is gimbal-locked to ±π/2** — scipy's `as_euler("xyz")` clamps the middle angle. The XYZ convention is kept for parity with Motive's UI display, but for any control or feedback that needs full-range yaw (e.g. body-frame rotation, yaw P-controller), use **`MQTTRigidBody.yaw`** instead. It's computed directly from the quaternion via `atan2` and gives a continuous angle in `[-π, +π]` regardless of roll/pitch.

## Sim vs Hardware Control Interface

MPC (`mpc_landing/mpc.py`) outputs accelerations `[ax, ay, az]`. The translation to actuator commands differs between sim and hardware:

| | Crazyflow sim | Crazyflie hardware (cflib) |
|---|---|---|
| API | `sim.attitude_control([roll, pitch, yaw, thrust])` | `cf.commander.send_setpoint(roll, pitch, yawrate, thrust)` |
| Attitude units | Radians | Degrees |
| Yaw | Absolute angle (rad) | **Rate** (deg/s) |
| Thrust | Newtons | PWM (0–65535) |
| Max tilt | 0.5 rad (~28.6°) | 15° (hw-teleop default) |
| State source | Perfect physics (zero noise/latency) | OptiTrack via MQTT (noisy velocity from finite diff, network latency) |
| Coordinate mapping | `cf_to_mpc_state()` in `sim/mpc_controller.py` | `optitrack_to_mpc_state()` in `hardware/mpc_teleop.py` |
| Roll/pitch axes | Swapped — see note below | Standard: +pitch = forward, -roll = left |

**Crazyflow roll/pitch visual rotation:** The Crazyflie 3D model in Crazyflow is visually rotated 90° from cflib's convention. The attitude controller itself uses **standard convention** (roll=lateral, pitch=forward in world frame).

- **teleop.py** (human-in-the-loop): Must swap pitch/roll to compensate for the visual model rotation so keyboard controls match what the pilot sees on screen:
  ```
  CF_roll  = radians(cflib_pitch)     # forward/back
  CF_pitch = -radians(cflib_roll)     # left/right
  ```
  See `sim/teleop.py` lines 152-153.
- **MPC** (world-frame control): Uses standard physics (roll=lateral, pitch=forward) and is **not affected** by the visual rotation. `mpc_accel_to_attitude()` in `sim/mpc_controller.py` is correct as-is — no swap needed.

Newton-to-PWM thrust calibration is not yet established.
