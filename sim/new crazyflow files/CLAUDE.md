# crazyflow-test

Crazyflie drone teleoperation demo using the [crazyflow](https://github.com/utiasDSL/crazyflow) simulation framework.

## Setup

Requires Python 3.13+. Uses [uv](https://docs.astral.sh/uv/) for package management.

```bash
uv sync          # Install dependencies
uv run teleop.py # Run the teleop controller
```

## Dependencies

- **crazyflow** (v0.1.0b0) - Drone simulation framework (installed from GitHub source)
- **pynput** - Keyboard input handling

## Project Structure

Single-file application: `teleop.py` contains the full teleop controller.

## Code Overview

The teleop loop runs at 500 Hz simulation frequency with ~60 FPS rendering. It uses attitude control mode (`attitude`) with RK4 integration and `so_rpy_rotor_drag` physics.

**Keyboard mapping:**
- W/S - pitch forward/backward
- A/D - roll left/right
- Shift/Z - increase/decrease thrust
- Q/E - yaw left/right
- R - reset to hover
- ESC - quit

**Key constants:**
- `MAX_ROLL` / `MAX_PITCH`: 0.5 rad
- `YAW_SPEED`: 1.0 rad/s
- `RAMP_SPEED`: 3.0 rad/s (smooth control ramping)
- Hover thrust derived from drone mass × gravity
