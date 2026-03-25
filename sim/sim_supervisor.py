"""
Simulation supervisor: closes the MPC loop over CoppeliaSim.

Replaces the real supervisor's mocap + serial pipeline with:
    CoppeliaSim state -> MPC -> accel_to_pwm -> CoppeliaSim PWM signals

No ROS, no serial, no CRSF — just MPC + CoppeliaSim ZMQ RemoteAPI.

Usage:
    cd sim/
    python sim_supervisor.py [target_x target_y target_z]

Defaults to hovering at [0.0, 0.5, 0.0] (0.5m altitude, centered).
"""

import sys
import os
import time
import numpy as np

# Add parent directory so we can import mpc_landing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from mpc_landing.mpc import MPCController, MPCConfig
from mpc_landing.reference import static_reference
from mpc_landing.conversions import accel_to_pwm

# ---------------------------------------------------------------------------
# Physical parameters (must match Lua controller and real drone)
# ---------------------------------------------------------------------------
MASS = 0.025            # kg (Air65 ~25g)
G = 9.81
MAX_ANGLE = 35 * 3.14159265 / 180  # rad (Betaflight default)
MAX_THRUST = 0.49       # N (~2:1 TWR)
DT = 0.02               # 50 Hz control loop


def get_drone_state(sim, drone_handle):
    """Read drone pose and velocity from CoppeliaSim.

    Returns:
        x0: np.array [px, vx, py, vy, pz, vz] matching MPC state convention
    """
    pos = sim.getObjectPosition(drone_handle, sim.handle_world)
    lin_vel, _ = sim.getObjectVelocity(drone_handle)
    return np.array([
        pos[0], lin_vel[0],   # X (forward) — position, velocity
        pos[2], lin_vel[2],   # Y (up) — CoppeliaSim Z is up
        pos[1], lin_vel[1],   # Z (lateral) — CoppeliaSim Y
    ])


def send_pwm(sim, roll, pitch, throttle, yaw):
    """Send PWM commands to CoppeliaSim float signals."""
    sim.setFloatSignal('cmd_roll', float(roll))
    sim.setFloatSignal('cmd_pitch', float(pitch))
    sim.setFloatSignal('cmd_thrust', float(throttle))
    sim.setFloatSignal('cmd_yaw', float(yaw))


def main():
    # Parse optional target position
    if len(sys.argv) == 4:
        target = [float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])]
    else:
        target = [0.0, 0.5, 0.0]  # default: hover at 0.5m altitude

    print(f"Target position: {target}")
    print(f"MPC config: dt={DT}, mass={MASS}kg, max_angle={MAX_ANGLE*180/3.14159:.0f}deg")

    # Connect to CoppeliaSim
    client = RemoteAPIClient()
    sim = client.getObject('sim')

    # Get drone object handle (adjust path to match your scene)
    drone_handle = sim.getObject('/Quadcopter')

    # Initialize MPC
    mpc = MPCController(MPCConfig(dt=DT))
    N = mpc.cfg.horizon

    # Start simulation in stepping mode for deterministic timing
    sim.setStepping(True)
    sim.startSimulation()

    # Send neutral PWM before first MPC solve
    send_pwm(sim, 1500, 1500, 1000, 1500)
    sim.step()

    print("\nRunning MPC loop (Ctrl+C to stop)...")
    print(f"{'step':>6}  {'px':>7} {'vx':>7} {'py':>7} {'vy':>7} {'pz':>7} {'vz':>7}  "
          f"{'roll':>6} {'pitch':>6} {'thr':>6} {'yaw':>6}")
    print("-" * 90)

    step = 0
    try:
        while sim.getSimulationState() != sim.simulation_stopped:
            # Read current state
            x0 = get_drone_state(sim, drone_handle)

            # Generate reference trajectory
            ref = static_reference(target, N, DT)

            # Solve MPC
            u = mpc.compute(x0, ref)
            ax, ay, az = u[0], u[1], u[2]

            # Convert accelerations to PWM
            roll_pwm, pitch_pwm, thr_pwm, yaw_pwm = accel_to_pwm(
                ax, ay, az,
                mass=MASS, g=G,
                max_angle=MAX_ANGLE,
                max_thrust=MAX_THRUST,
            )

            # Send to sim
            send_pwm(sim, roll_pwm, pitch_pwm, thr_pwm, yaw_pwm)

            # Print telemetry every 10 steps (5 Hz)
            if step % 10 == 0:
                print(f"{step:6d}  "
                      f"{x0[0]:+7.3f} {x0[1]:+7.3f} {x0[2]:+7.3f} "
                      f"{x0[3]:+7.3f} {x0[4]:+7.3f} {x0[5]:+7.3f}  "
                      f"{roll_pwm:6.0f} {pitch_pwm:6.0f} {thr_pwm:6.0f} {yaw_pwm:6.0f}")

            # Advance simulation by one step
            sim.step()
            step += 1

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        send_pwm(sim, 1500, 1500, 1000, 1500)
        sim.stop()
        print("Simulation stopped.")


if __name__ == "__main__":
    main()
