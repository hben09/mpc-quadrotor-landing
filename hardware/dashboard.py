"""
Real-time 3D viewer for Crazyflie drone position via MQTT.

Subscribes to OptiTrack rigid-body data over MQTT and displays the drone
(and optionally the landing target) in a PyVista 3D scene with
RASTIC arena boundaries and a flight trail.

Usage:
    uv run dashboard
    uv run dashboard --broker localhost
"""

import argparse
import json
import threading

import numpy as np
import paho.mqtt.client as mqtt
import pyvista as pv
import vtk
from scipy.spatial.transform import Rotation

from mpc_landing.boundary import ARENA_BOUNDS
from mpc_landing.mqtt.parser import MQTTRigidBody, RigidBodyTracker

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BROKER = "rasticvm.internal"
PORT = 1883
DRONE_TOPIC = "rb/crazyflie"
LANDING_TOPIC = "rb/landing"
MPC_TARGET_TOPIC = "mpc/target"
MPC_TRAJ_TOPIC = "mpc/trajectory"
MPC_REF_TOPIC = "mpc/reference"
MPC_CONE_TOPIC = "mpc/cone"
BATTERY_TOPIC = "cf/battery"

BATTERY_STATE_LABELS = {0: "OK", 1: "CHG", 2: "LOW", 3: "SHUT"}
CALLBACK_MS = 50  # 20 Hz
MAX_TRAIL_POINTS = 2000
DRONE_RADIUS = 0.05
LANDING_RADIUS = 0.10
AXIS_LENGTH = 0.15


# ---------------------------------------------------------------------------
# Thread-safe shared state
# ---------------------------------------------------------------------------
class TrackedState:
    def __init__(self):
        self._lock = threading.Lock()
        self._drone: MQTTRigidBody | None = None
        self._landing: MQTTRigidBody | None = None
        self._mpc_target: list[float] | None = None
        self._mpc_traj: list[list[float]] | None = None
        self._mpc_ref: list[list[float]] | None = None
        self._mpc_cone: dict | None = None
        self._battery: dict | None = None

    def set_drone(self, rb: MQTTRigidBody):
        with self._lock:
            self._drone = rb

    def set_landing(self, rb: MQTTRigidBody):
        with self._lock:
            self._landing = rb

    def set_mpc_target(self, pos: list[float]):
        with self._lock:
            self._mpc_target = pos

    def set_mpc_traj(self, points: list[list[float]]):
        with self._lock:
            self._mpc_traj = points

    def set_mpc_ref(self, points: list[list[float]]):
        with self._lock:
            self._mpc_ref = points

    def set_mpc_cone(self, cone: dict | None):
        with self._lock:
            self._mpc_cone = cone

    def set_battery(self, battery: dict):
        with self._lock:
            self._battery = battery

    def get(self) -> tuple[MQTTRigidBody | None, MQTTRigidBody | None]:
        with self._lock:
            return self._drone, self._landing

    def get_mpc(
        self,
    ) -> tuple[
        list[float] | None,
        list[list[float]] | None,
        list[list[float]] | None,
        dict | None,
    ]:
        with self._lock:
            return self._mpc_target, self._mpc_traj, self._mpc_ref, self._mpc_cone

    def get_battery(self) -> dict | None:
        with self._lock:
            return self._battery


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------
def mqtt_to_pyvista(pos):
    """MQTT [x, y, z] (x=forward, y=altitude, z=lateral) -> PyVista (z-up)."""
    return (pos[0], -pos[2], pos[1])


# ---------------------------------------------------------------------------
# Arena wireframe (same remap as mpc_gui.py)
# ---------------------------------------------------------------------------
def build_arena_wireframe():
    """RASTIC arena as a PyVista wireframe box."""
    b = ARENA_BOUNDS
    return pv.Box(
        bounds=[
            b["x_min"],
            b["x_max"],
            -b["z_max"],
            -b["z_min"],
            b["y_min"],
            b["y_max"],
        ]
    )


# ---------------------------------------------------------------------------
# Trail mesh
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
# Orientation axes
# ---------------------------------------------------------------------------
def build_orientation_axes(pv_pos, rot_quat):
    """Build RGB body-frame axes from drone position and quaternion.

    Returns a PolyData with 3 line segments (red=x, green=y, blue=z)
    in PyVista coordinates.
    """
    R = Rotation.from_quat(rot_quat).as_matrix()  # 3x3 in MQTT frame

    # Body axes in MQTT frame, then convert each endpoint to PyVista
    origin = np.array(pv_pos)
    points = [origin]
    lines = []
    colors = []

    for i, color in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
        # Axis direction in MQTT frame [dx, dy, dz], remap to PyVista (x, z, y)
        mqtt_dir = R[:, i] * AXIS_LENGTH
        pv_endpoint = (
            pv_pos[0] + mqtt_dir[0],
            pv_pos[1] - mqtt_dir[2],
            pv_pos[2] + mqtt_dir[1],
        )
        points.append(pv_endpoint)
        idx = len(points) - 1
        lines.extend([2, 0, idx])
        colors.append(color)

    pts = np.array(points)
    mesh = pv.PolyData(pts, lines=np.array(lines))
    # Assign per-cell colors
    mesh.cell_data["colors"] = np.array(colors, dtype=np.uint8)
    return mesh


# ---------------------------------------------------------------------------
# MQTT client
# ---------------------------------------------------------------------------
def start_mqtt(state: TrackedState, broker: str, port: int):
    drone_tracker = RigidBodyTracker()
    landing_tracker = RigidBodyTracker()

    def on_connect(client, userdata, flags, reason_code, properties):
        print(f"MQTT connected (rc={reason_code})")
        client.subscribe(DRONE_TOPIC)
        client.subscribe(LANDING_TOPIC)
        client.subscribe(MPC_TARGET_TOPIC)
        client.subscribe(MPC_TRAJ_TOPIC)
        client.subscribe(MPC_REF_TOPIC)
        client.subscribe(MPC_CONE_TOPIC)
        client.subscribe(BATTERY_TOPIC)
        print(
            f"Subscribed to '{DRONE_TOPIC}', '{LANDING_TOPIC}', 'mpc/#', '{BATTERY_TOPIC}'"
        )

    def on_message(client, userdata, msg):
        try:
            if msg.topic == DRONE_TOPIC:
                rb = drone_tracker.update(msg.payload.decode())
                state.set_drone(rb)
            elif msg.topic == LANDING_TOPIC:
                rb = landing_tracker.update(msg.payload.decode())
                state.set_landing(rb)
            elif msg.topic == MPC_TARGET_TOPIC:
                data = json.loads(msg.payload.decode())
                state.set_mpc_target(data["pos"])
            elif msg.topic == MPC_TRAJ_TOPIC:
                data = json.loads(msg.payload.decode())
                state.set_mpc_traj(data["points"])
            elif msg.topic == MPC_REF_TOPIC:
                data = json.loads(msg.payload.decode())
                state.set_mpc_ref(data["points"])
            elif msg.topic == MPC_CONE_TOPIC:
                data = json.loads(msg.payload.decode())
                state.set_mpc_cone(data if data.get("apex") is not None else None)
            elif msg.topic == BATTERY_TOPIC:
                state.set_battery(json.loads(msg.payload.decode()))
        except Exception as e:
            print(f"Parse error on {msg.topic}: {e}")

    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id="mpc-quadrotor-viewer",
        protocol=mqtt.MQTTv311,
    )
    client.on_connect = on_connect
    client.on_message = on_message

    print(f"Connecting to {broker}:{port} ...")
    client.connect(broker, port)
    client.loop_start()
    return client


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MQTT 3D Drone Viewer")
    parser.add_argument(
        "--broker", default=BROKER, help=f"MQTT broker (default: {BROKER})"
    )
    parser.add_argument(
        "--port", type=int, default=PORT, help=f"MQTT port (default: {PORT})"
    )
    args = parser.parse_args()

    state = TrackedState()
    client = start_mqtt(state, args.broker, args.port)

    # --- PyVista plotter ---
    plotter = pv.Plotter(title="MQTT Drone Viewer")
    plotter.set_background("white")

    # Arena wireframe
    arena = build_arena_wireframe()
    plotter.add_mesh(arena, style="wireframe", color="gray", line_width=1)

    # Ground plane
    ground = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), i_size=10, j_size=10)
    plotter.add_mesh(ground, color="lightgray", opacity=0.3)

    # Drone sphere
    drone_mesh = pv.Sphere(radius=DRONE_RADIUS, center=(0, 0, 0))
    plotter.add_mesh(drone_mesh, color="blue", name="drone")

    # Landing sphere
    landing_mesh = pv.Sphere(radius=LANDING_RADIUS, center=(0, 0, 0))
    plotter.add_mesh(landing_mesh, color="yellow", name="landing", opacity=0.7)

    # MPC target sphere
    target_mesh = pv.Sphere(radius=0.06, center=(0, 0, 0))
    plotter.add_mesh(target_mesh, color="red", name="mpc_target", opacity=0.0)

    # Orientation axes (added on first data arrival)

    # Trail state
    trail_points: list[tuple[float, float, float]] = []

    # HUD
    plotter.add_text(
        "Waiting for MQTT data...", position="upper_right", font_size=10, name="hud"
    )

    # Camera and axes
    plotter.camera_position = "xz"
    # Rotate the widget 90° about X so red/green/blue arrows align with
    # MQTT X/Y/Z (fwd/up/right) instead of PyVista's remapped frame.
    axes_actor = plotter.add_axes(
        xlabel="X",
        ylabel="Y",
        zlabel="Z",
        x_color="red",
        y_color="green",
        z_color="blue",
    )
    _axes_transform = vtk.vtkTransform()
    _axes_transform.RotateX(90)
    axes_actor.SetUserTransform(_axes_transform)

    # --- Timer callback (20 Hz) ---
    def update_callback(_step=None):
        drone, landing = state.get()
        mpc_target, mpc_traj, mpc_ref, mpc_cone = state.get_mpc()

        # Update drone
        if drone is not None:
            pv_pos = mqtt_to_pyvista(drone.pos)
            drone_mesh.copy_from(pv.Sphere(radius=DRONE_RADIUS, center=pv_pos))

            try:
                axes = build_orientation_axes(pv_pos, drone.rot)
                plotter.add_mesh(
                    axes, scalars="colors", rgb=True, line_width=3, name="axes"
                )
            except Exception:
                pass  # Skip if quaternion is degenerate

            trail_points.append(pv_pos)
            if len(trail_points) > MAX_TRAIL_POINTS:
                trail_points.pop(0)
            if len(trail_points) >= 2:
                trail = build_trail_mesh(trail_points)
                plotter.add_mesh(trail, color="green", line_width=2, name="trail")

        # Update landing
        if landing is not None:
            landing_pv = mqtt_to_pyvista(landing.pos)
            landing_mesh.copy_from(pv.Sphere(radius=LANDING_RADIUS, center=landing_pv))
            try:
                landing_axes = build_orientation_axes(landing_pv, landing.rot)
                plotter.add_mesh(
                    landing_axes,
                    scalars="colors",
                    rgb=True,
                    line_width=3,
                    name="landing_axes",
                )
            except Exception:
                pass

        # Update MPC target + trajectory
        if mpc_target is not None:
            tgt_pv = mqtt_to_pyvista(mpc_target)
            target_mesh.copy_from(pv.Sphere(radius=0.06, center=tgt_pv))
            plotter.add_mesh(target_mesh, color="red", name="mpc_target", opacity=0.8)
        if mpc_traj is not None and len(mpc_traj) >= 2:
            traj_pv = [mqtt_to_pyvista(p) for p in mpc_traj]
            traj_line = build_trail_mesh(traj_pv)
            plotter.add_mesh(traj_line, color="orange", line_width=3, name="mpc_traj")
        if mpc_ref is not None and len(mpc_ref) >= 2:
            ref_pv = [mqtt_to_pyvista(p) for p in mpc_ref]
            ref_line = build_trail_mesh(ref_pv)
            plotter.add_mesh(ref_line, color="magenta", line_width=2, name="mpc_ref")
            ref_points = pv.PolyData(np.array(ref_pv))
            plotter.add_mesh(
                ref_points,
                color="magenta",
                render_points_as_spheres=True,
                point_size=8,
                name="mpc_ref_pts",
            )

        # Landing approach frustum (20 cm landing disk at pad, opens upward)
        if mpc_cone is not None:
            apex_mqtt = mpc_cone["apex"]
            max_height = float(mpc_cone["max_height"])
            half_angle_deg = float(mpc_cone["half_angle_deg"])
            base_radius = float(mpc_cone.get("base_radius", 0.0))
            height = max(max_height - float(apex_mqtt[1]), 1e-3)
            top_radius = base_radius + height * np.tan(np.radians(half_angle_deg))
            apex_pv = mqtt_to_pyvista(apex_mqtt)
            cone_color = "cyan" if mpc_cone.get("inside") else "orange"

            n = 32
            theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            bottom_z = apex_pv[2]
            top_z = apex_pv[2] + height
            bottom_ring = np.column_stack(
                [
                    apex_pv[0] + base_radius * cos_t,
                    apex_pv[1] + base_radius * sin_t,
                    np.full(n, bottom_z),
                ]
            )
            top_ring = np.column_stack(
                [
                    apex_pv[0] + top_radius * cos_t,
                    apex_pv[1] + top_radius * sin_t,
                    np.full(n, top_z),
                ]
            )
            points = np.vstack([bottom_ring, top_ring])

            lines: list[int] = []
            for i in range(n):
                lines.extend([2, i, (i + 1) % n])
            for i in range(n):
                lines.extend([2, n + i, n + ((i + 1) % n)])
            for i in range(0, n, 4):
                lines.extend([2, i, n + i])

            frustum = pv.PolyData(points, lines=np.array(lines))
            plotter.add_mesh(
                frustum,
                style="wireframe",
                color=cone_color,
                opacity=0.5,
                line_width=1,
                name="mpc_cone",
            )
        else:
            plotter.remove_actor("mpc_cone", render=False)

        # Update HUD
        mpc_line = "MPC: --"
        if mpc_target is not None:
            t = mpc_target
            mpc_line = f"MPC tgt: [{t[0]:+.3f}, {t[1]:+.3f}, {t[2]:+.3f}]"
        if drone is not None:
            drone_lines = (
                f"pos: [{drone.pos[0]:+.3f}, {drone.pos[1]:+.3f}, {drone.pos[2]:+.3f}]\n"
                f"vel: [{drone.vel[0]:+.3f}, {drone.vel[1]:+.3f}, {drone.vel[2]:+.3f}]\n"
                f"euler: [{drone.euler[0]:+.2f}, {drone.euler[1]:+.2f}, {drone.euler[2]:+.2f}]\n"
            )
        else:
            drone_lines = "drone: --\n"
        battery = state.get_battery()
        if battery is not None:
            label = BATTERY_STATE_LABELS.get(battery["state"], f"?{battery['state']}")
            battery_line = f"bat: {battery['vbat']:.2f}V {battery['level']}% [{label}]"
        else:
            battery_line = "bat: --"
        plotter.add_text(
            f"{drone_lines}{mpc_line}\n{battery_line}",
            position="upper_right",
            font_size=10,
            name="hud",
        )

    plotter.add_timer_event(
        max_steps=999_999, duration=CALLBACK_MS, callback=update_callback
    )

    print("\n=== MQTT Drone Viewer ===")
    print(f"Broker: {args.broker}:{args.port}")
    print(f"Topics: {DRONE_TOPIC}, {LANDING_TOPIC}")
    print("Close the window to exit.")
    print("=" * 26)

    plotter.show()
    client.loop_stop()
    client.disconnect()
    print("Done.")


if __name__ == "__main__":
    main()
