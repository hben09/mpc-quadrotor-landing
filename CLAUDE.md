# CLAUDE.md

## Project Overview

MPC-based autonomous quadrotor landing system. A Crazyflie drone is controlled from a PC via Crazyradio 2.0, with state feedback from an OptiTrack motion capture system streamed over MQTT.

## Repository Structure

```
src/                  # Main control package
  mpc.py              # Linear MPC controller (CVXPY/OSQP, 3D double integrator)
  pid.py              # PID baseline controller (currently active in supervisor)
  reference.py        # Reference trajectory generation (tracking, landing, static)
  boundary.py         # RASTIC arena boundary safety checker
  mqtt/               # MQTT rigid-body pose streaming from OptiTrack
    parser.py         # Reusable parser: JSON → MQTTRigidBody dataclass with velocity
    sub.py            # MQTT subscriber for drone (crazyflie) and ground vehicle (limo777) poses

sim/                  # Crazyflow simulation environment
  teleop.py           # Keyboard teleoperation (attitude control, 500Hz, pynput)
  old/                # Legacy CoppeliaSim scripts (archived)
```

## Architecture

### Control Pipeline
1. **Drone pose** → OptiTrack → Motive → MQTT broker (`rasticvm.internal:1883`) → topic `rb/crazyflie` → `mqtt_parser.py` → `MQTTRigidBody` (pos, euler, vel)
2. **Ground vehicle pose** → OptiTrack → Motive → MQTT broker → topic `rb/limo777` → `mqtt_parser.py` → `MQTTRigidBody` (pos, euler, vel)
3. **Controller** (PID or MPC) computes desired commands (control interface TBD)
4. **cflib** → Crazyradio 2.0 → Crazyflie

### Current State
- `supervisor.py` uses **PID** controller — MPC is implemented in `mpc.py` but not yet integrated
- MPC is a drop-in replacement: call `mpc_controller.compute(x0, reference)` instead of `pid.compute(target)`
- MPC state: [px, vx, py, vy, pz, vz], control: [ax, ay, az], horizon: 25 steps (0.5s)

## Dependencies

- **Python**: numpy, cvxpy, scipy, cflib, paho-mqtt
- **Sim**: crazyflow (from GitHub, managed via uv in sim/), pynput

## Hardware

- Drone: Crazyflie 2.1 (Crazyflie firmware)
- Crazyradio 2.0 USB dongle connected to PC
- OptiTrack motion capture system (3 markers per drone)
- RASTIC arena bounds: X [-4.5, 3.0], Y [0.0, 2.0], Z [-2.0, 3.0] meters

## Key Conventions

- Control loop: 50Hz (20ms period)
- Crazyflie control interface: `cf.commander.send_setpoint(roll, pitch, yawrate, thrust)` — attitude commands via cflib
