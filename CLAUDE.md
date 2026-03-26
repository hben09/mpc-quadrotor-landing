# CLAUDE.md

## Project Overview

MPC-based autonomous quadrotor landing system. A drone (TinyWhoop Air65 on Betaflight) is controlled from a PC via ELRS wireless link, with state feedback from an OptiTrack motion capture system.

## Repository Structure

```
src/                  # Main control package
  supervisor.py       # Main ROS 2 node — 50Hz control loop, mocap + serial I/O
  mpc.py              # Linear MPC controller (CVXPY/OSQP, 3D double integrator)
  pid.py              # PID baseline controller (currently active in supervisor)
  reference.py        # Reference trajectory generation (tracking, landing, static)
  boundary.py         # RASTIC arena boundary safety checker
  crsf.py             # CRSF protocol encoding for ELRS TX module
  mqtt_sub.py         # MQTT subscriber for ground vehicle (limo777) pose
  mqtt_parser.py      # Reusable parser: JSON → MQTTRigidBody dataclass with velocity

sim/                  # CoppeliaSim simulation environment
  keyboard_teleop.py  # Manual keyboard control via ZMQ RemoteAPI
  my sim.ttt          # CoppeliaSim scene file
  *.lua               # Drone and ground platform controller scripts
```

## Architecture

### Control Pipeline
1. **OptiTrack mocap** → Motive → VRPN → ROS 2 `/vrpn_mocap/AzamatDrone/pose` → supervisor gets (x, y, z, roll, pitch, yaw)
2. **Ground vehicle pose** → OptiTrack → Motive → MQTT `rb/limo777` → `mqtt_sub.py` → `mqtt_parser.py` → `MQTTRigidBody` (pos, euler, vel)
3. **Controller** (PID or MPC) computes desired roll, pitch, yaw, throttle as PWM (1000–2000 µs)
4. **CRSF encoding** (`crsf.py`) packs into 26-byte frames
5. **Serial** (pyserial, 115200 baud, `/dev/ttyACM0`) → ELRS TX module → drone's Betaflight FC

### Current State
- `supervisor.py` uses **PID** controller — MPC is implemented in `mpc.py` but not yet integrated
- MPC is a drop-in replacement: call `mpc_controller.compute(x0, reference)` instead of `pid.compute(target)`
- MPC state: [px, vx, py, vy, pz, vz], control: [ax, ay, az], horizon: 25 steps (0.5s)

## Dependencies

- **ROS 2 Humble**: rclpy, geometry_msgs, sensor_msgs
- **Python**: numpy, cvxpy, scipy, pyserial, paho-mqtt
- **Sim**: coppeliasim-zmqremoteapi-client (managed via uv in sim/)

## Hardware

- Drone: TinyWhoop Air65 with Betaflight + ELRS receiver
- ELRS TX module connected to PC via USB
- OptiTrack motion capture system (3 markers per drone)
- RASTIC arena bounds: X [-4.5, 3.0], Y [0.0, 2.0], Z [-2.0, 3.0] meters

## Key Conventions

- PWM range: 1000–2000 µs (1500 = center for roll/pitch/yaw, 1000 = zero throttle)
- CRSF channel order: roll(0), pitch(1), throttle(2), yaw(3), arm(4), flight mode(5)
- Control loop: 50Hz (20ms period)
- Flight modes: ACRO (rate commands) or ANGLE/STAB (angle commands, self-leveling)
