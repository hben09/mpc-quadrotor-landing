"""
PyVista 3D GUI for MPC quadrotor control in Crazyflow.

Drag the red target sphere to control where the drone flies. Displays
the MPC planned trajectory, actual flight path, and RASTIC arena bounds.

Usage:
    uv run sim-mpc-gui
"""

import numpy as np
import pyvista as pv

from crazyflow.control import Control
from crazyflow.sim import Physics, Sim
from crazyflow.sim.integration import Integrator

from mpc_landing import MPCController, MPCConfig, ARENA_BOUNDS, static_reference


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
G = 9.81
SIM_FREQ = 500
CONTROL_HZ = 50
CONTROL_DT = 1.0 / CONTROL_HZ
STEPS_PER_MPC = SIM_FREQ // CONTROL_HZ  # 10
MAX_TILT = 0.5  # rad
HOVER_ALTITUDE = 1.0
CALLBACK_MS = 20  # 50 Hz
MAX_TRAIL_POINTS = 2000


# ---------------------------------------------------------------------------
# Helpers (copied from mpc_controller.py — pure functions)
# ---------------------------------------------------------------------------
def cf_to_mpc_state(pos, vel):
    """Crazyflow (x, y, z) z-up -> MPC [px, vx, py, vy, pz, vz]."""
    return np.array([
        pos[0], vel[0],   # px, vx  (CF x = forward)
        pos[2], vel[2],   # py, vy  (CF z = altitude)
        pos[1], vel[1],   # pz, vz  (CF y = lateral)
    ])


def mpc_accel_to_attitude(ax, ay, az, mass, thrust_max, max_tilt=MAX_TILT):
    """MPC accelerations -> Crazyflow attitude command [roll, pitch, yaw, thrust]."""
    pitch = np.clip(ax / G, -max_tilt, max_tilt)
    roll = np.clip(-az / G, -max_tilt, max_tilt)
    yaw = 0.0
    thrust = np.clip(mass * (G + ay), 0.0, thrust_max)
    return np.array([roll, pitch, yaw, thrust])


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------
def mpc_traj_to_cf(traj):
    """MPC trajectory (N+1, 6) -> Crazyflow display coords (N+1, 3).

    MPC: [px, vx, py, vy, pz, vz]  ->  CF: [x=px, y=pz, z=py]
    """
    return np.column_stack([traj[:, 0], traj[:, 4], traj[:, 2]])


def cf_target_to_mpc(cf_pos):
    """Crazyflow [x, y, z] target -> MPC [px, py, pz] for static_reference."""
    return [cf_pos[0], cf_pos[2], cf_pos[1]]


# ---------------------------------------------------------------------------
# Arena
# ---------------------------------------------------------------------------
def build_arena_wireframe():
    """RASTIC arena as a PyVista wireframe box.

    ARENA_BOUNDS: x=forward, y=altitude, z=lateral
    PyVista Box: [xmin, xmax, ymin, ymax, zmin, zmax] with z-up
    Remap: arena y (altitude) -> pv z, arena z (lateral) -> pv y
    """
    b = ARENA_BOUNDS
    return pv.Box(bounds=[
        b['x_min'], b['x_max'],
        b['z_min'], b['z_max'],
        b['y_min'], b['y_max'],
    ])


# ---------------------------------------------------------------------------
# Trail mesh builder
# ---------------------------------------------------------------------------
def build_trail_mesh(points):
    """Build a polyline mesh from a list of 3D points."""
    pts = np.array(points)
    n = len(pts)
    lines = np.empty((n - 1) * 3, dtype=int)
    for i in range(n - 1):
        lines[i * 3] = 2
        lines[i * 3 + 1] = i
        lines[i * 3 + 2] = i + 1
    return pv.PolyData(pts, lines=lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # --- Crazyflow sim (headless — no sim.render()) ---
    print("Initializing Crazyflow sim...")
    sim = Sim(
        n_worlds=1,
        n_drones=1,
        physics=Physics.so_rpy_rotor_drag,
        control=Control.attitude,
        integrator=Integrator.rk4,
        freq=SIM_FREQ,
        attitude_freq=SIM_FREQ,
        device="cpu",
    )
    sim.reset()
    mass = float(sim.data.params.mass[0, 0, 0])
    thrust_max = float(sim.data.controls.attitude.params.thrust_max) * 4

    # --- MPC (mutable — sliders rebuild it) ---
    config = MPCConfig(dt=CONTROL_DT, horizon=25)
    mpc_state = {"mpc": MPCController(config), "config": config, "max_tilt": MAX_TILT}
    N = config.horizon
    print(f"MPC: horizon={N}, dt={CONTROL_DT}s, mass={mass:.4f}kg")

    def rebuild_mpc():
        mpc_state["mpc"] = MPCController(mpc_state["config"])

    # --- PyVista plotter ---
    plotter = pv.Plotter(title="MPC Quadrotor GUI")
    plotter.set_background("white")

    # Arena wireframe
    arena = build_arena_wireframe()
    plotter.add_mesh(arena, style="wireframe", color="gray", line_width=1)

    # Ground plane
    ground = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), i_size=10, j_size=10)
    plotter.add_mesh(ground, color="lightgray", opacity=0.3)

    # Drone sphere
    drone_mesh = pv.Sphere(radius=0.05, center=(0, 0, 0))
    plotter.add_mesh(drone_mesh, color="blue", name="drone")

    # MPC trajectory line (initialized with dummy points)
    init_pts = np.zeros((N + 1, 3))
    init_pts[:, 2] = np.linspace(0, 0.01, N + 1)  # slight offset to avoid degenerate line
    traj_mesh = pv.Spline(init_pts, n_points=N + 1)
    plotter.add_mesh(traj_mesh, color="orange", line_width=3, name="trajectory")

    # Path trail state
    trail_points = []
    trail_added = False

    # Draggable target
    target_cf = np.array([0.0, 0.0, HOVER_ALTITUDE])

    def target_callback(point):
        nonlocal target_cf
        target_cf = np.array(point)

    plotter.add_sphere_widget(
        target_callback,
        center=target_cf,
        radius=0.08,
        color="red",
        style="surface",
        test_callback=False,
    )

    # --- Parameter sliders ---
    def on_q_pos(value):
        c = mpc_state["config"]
        c.Q_diag = [value, 1.0, value, 1.0, value, 1.0]
        rebuild_mpc()

    def on_r(value):
        c = mpc_state["config"]
        c.R_diag = [value, value, value]
        rebuild_mpc()

    def on_a_max(value):
        mpc_state["config"].a_max = value
        rebuild_mpc()

    def on_v_max(value):
        mpc_state["config"].v_max = value
        rebuild_mpc()

    plotter.add_slider_widget(
        on_q_pos, rng=[1, 100], value=10, title="Q (position)",
        pointa=(0.02, 0.92), pointb=(0.28, 0.92),
        interaction_event="end",
    )
    plotter.add_slider_widget(
        on_r, rng=[0.01, 20], value=1.0, title="R (effort)",
        pointa=(0.02, 0.82), pointb=(0.28, 0.82),
        interaction_event="end",
    )
    plotter.add_slider_widget(
        on_a_max, rng=[1, 15], value=5.0, title="a_max (m/s^2)",
        pointa=(0.02, 0.72), pointb=(0.28, 0.72),
        interaction_event="end",
    )
    plotter.add_slider_widget(
        on_v_max, rng=[0.5, 5], value=2.0, title="v_max (m/s)",
        pointa=(0.02, 0.62), pointb=(0.28, 0.62),
        interaction_event="end",
    )

    def on_qf(value):
        c = mpc_state["config"]
        c.Qf_diag = [value, value / 10, value, value / 10, value, value / 10]
        rebuild_mpc()

    def on_q_vel(value):
        c = mpc_state["config"]
        c.Q_diag[1] = value  # vx
        c.Q_diag[3] = value  # vy
        c.Q_diag[5] = value  # vz
        rebuild_mpc()

    def on_max_tilt(value):
        mpc_state["max_tilt"] = value

    plotter.add_slider_widget(
        on_qf, rng=[10, 500], value=200, title="Qf (terminal)",
        pointa=(0.02, 0.52), pointb=(0.28, 0.52),
        interaction_event="end",
    )
    plotter.add_slider_widget(
        on_q_vel, rng=[0.1, 20], value=1.0, title="Q (velocity)",
        pointa=(0.02, 0.42), pointb=(0.28, 0.42),
        interaction_event="end",
    )
    plotter.add_slider_widget(
        on_max_tilt, rng=[0.1, 1.0], value=0.5, title="max_tilt (rad)",
        pointa=(0.02, 0.32), pointb=(0.28, 0.32),
        interaction_event="end",
    )

    # Camera and axes
    plotter.camera_position = "xz"
    plotter.add_axes()

    # --- Control callback (50 Hz) ---
    def control_callback(_step=None):
        nonlocal trail_added

        # 1. Read drone state
        pos_cf = np.array(sim.data.states.pos[0, 0])
        vel_cf = np.array(sim.data.states.vel[0, 0])
        x0 = cf_to_mpc_state(pos_cf, vel_cf)

        # 2. Build reference from dragged target
        mpc_target = cf_target_to_mpc(target_cf)
        mpc = mpc_state["mpc"]
        ref = static_reference(mpc_target, N, CONTROL_DT)

        # 3. Solve MPC
        u_opt = mpc.compute(x0, ref)
        ax, ay, az = u_opt

        # 4. Apply to sim
        attitude = mpc_accel_to_attitude(ax, ay, az, mass, thrust_max, mpc_state["max_tilt"])
        cmd = np.zeros((1, 1, 4))
        cmd[0, 0, :] = attitude
        sim.attitude_control(cmd)
        sim.step(STEPS_PER_MPC)

        # 5. Update drone position
        new_pos = np.array(sim.data.states.pos[0, 0])
        drone_mesh.copy_from(pv.Sphere(radius=0.05, center=tuple(new_pos)))

        # 6. Update MPC planned trajectory
        planned = mpc_state["mpc"].get_planned_trajectory()
        if planned is not None:
            cf_traj = mpc_traj_to_cf(np.array(planned))
            if len(cf_traj) >= 2:
                traj_mesh.copy_from(pv.Spline(cf_traj, n_points=len(cf_traj)))

        # 7. Update path trail
        trail_points.append(new_pos.tolist())
        if len(trail_points) > MAX_TRAIL_POINTS:
            trail_points.pop(0)
        if len(trail_points) >= 2:
            trail = build_trail_mesh(trail_points)
            if not trail_added:
                plotter.add_mesh(trail, color="green", line_width=2, name="trail")
                trail_added = True
            else:
                plotter.add_mesh(trail, color="green", line_width=2, name="trail")

    plotter.add_timer_event(
        max_steps=999_999, duration=CALLBACK_MS, callback=control_callback
    )

    print("\n=== MPC Quadrotor GUI ===")
    print("Drag the red sphere to set the target.")
    print("Close the window to exit.")
    print("=" * 28)

    plotter.show()
    sim.close()
    print("Done.")


if __name__ == "__main__":
    main()
