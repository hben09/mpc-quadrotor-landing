"""
Real-time 3D viewer for Crazyflie drone position via MQTT.

Subscribes to OptiTrack rigid-body data over MQTT and displays the drone
(and optionally the limo777 ground vehicle) in a PyVista 3D scene with
RASTIC arena boundaries and a flight trail.

Usage:
    uv run mqtt-viewer
    uv run mqtt-viewer --broker localhost
"""

import argparse
import threading

import numpy as np
import paho.mqtt.client as mqtt
import pyvista as pv
from scipy.spatial.transform import Rotation

from mpc_landing.boundary import ARENA_BOUNDS
from mpc_landing.mqtt.parser import MQTTRigidBody, RigidBodyTracker

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BROKER = "rasticvm.internal"
PORT = 1883
DRONE_TOPIC = "rb/crazyflie"
LIMO_TOPIC = "rb/limo777"
CALLBACK_MS = 50  # 20 Hz
MAX_TRAIL_POINTS = 2000
DRONE_RADIUS = 0.05
LIMO_RADIUS = 0.08
AXIS_LENGTH = 0.15


# ---------------------------------------------------------------------------
# Thread-safe shared state
# ---------------------------------------------------------------------------
class TrackedState:
    def __init__(self):
        self._lock = threading.Lock()
        self._drone: MQTTRigidBody | None = None
        self._limo: MQTTRigidBody | None = None

    def set_drone(self, rb: MQTTRigidBody):
        with self._lock:
            self._drone = rb

    def set_limo(self, rb: MQTTRigidBody):
        with self._lock:
            self._limo = rb

    def get(self) -> tuple[MQTTRigidBody | None, MQTTRigidBody | None]:
        with self._lock:
            return self._drone, self._limo


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------
def mqtt_to_pyvista(pos):
    """MQTT [x, y, z] (x=forward, y=altitude, z=lateral) -> PyVista (z-up)."""
    return (pos[0], pos[2], pos[1])


# ---------------------------------------------------------------------------
# Arena wireframe (same remap as mpc_gui.py)
# ---------------------------------------------------------------------------
def build_arena_wireframe():
    """RASTIC arena as a PyVista wireframe box."""
    b = ARENA_BOUNDS
    return pv.Box(bounds=[
        b['x_min'], b['x_max'],
        b['z_min'], b['z_max'],
        b['y_min'], b['y_max'],
    ])


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
            pv_pos[1] + mqtt_dir[2],
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
    limo_tracker = RigidBodyTracker()

    def on_connect(client, userdata, flags, reason_code, properties):
        print(f"MQTT connected (rc={reason_code})")
        client.subscribe(DRONE_TOPIC)
        client.subscribe(LIMO_TOPIC)
        print(f"Subscribed to '{DRONE_TOPIC}', '{LIMO_TOPIC}'")

    def on_message(client, userdata, msg):
        try:
            if msg.topic == DRONE_TOPIC:
                rb = drone_tracker.update(msg.payload.decode())
                state.set_drone(rb)
            elif msg.topic == LIMO_TOPIC:
                rb = limo_tracker.update(msg.payload.decode())
                state.set_limo(rb)
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
    parser.add_argument("--broker", default=BROKER, help=f"MQTT broker (default: {BROKER})")
    parser.add_argument("--port", type=int, default=PORT, help=f"MQTT port (default: {PORT})")
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

    # Limo sphere
    limo_mesh = pv.Sphere(radius=LIMO_RADIUS, center=(0, 0, 0))
    plotter.add_mesh(limo_mesh, color="red", name="limo", opacity=0.6)

    # Orientation axes (added on first data arrival)

    # Trail state
    trail_points: list[tuple[float, float, float]] = []

    # HUD
    plotter.add_text("Waiting for MQTT data...", position="upper_right", font_size=10, name="hud")

    # Camera and axes
    plotter.camera_position = "xz"
    plotter.add_axes()

    # --- Timer callback (20 Hz) ---
    def update_callback(_step=None):
        drone, limo = state.get()

        if drone is None:
            return

        # Update drone sphere
        pv_pos = mqtt_to_pyvista(drone.pos)
        drone_mesh.copy_from(pv.Sphere(radius=DRONE_RADIUS, center=pv_pos))

        # Update orientation axes
        try:
            axes = build_orientation_axes(pv_pos, drone.rot)
            plotter.add_mesh(axes, scalars="colors", rgb=True, line_width=3, name="axes")
        except Exception:
            pass  # Skip if quaternion is degenerate

        # Update trail
        trail_points.append(pv_pos)
        if len(trail_points) > MAX_TRAIL_POINTS:
            trail_points.pop(0)
        if len(trail_points) >= 2:
            trail = build_trail_mesh(trail_points)
            plotter.add_mesh(trail, color="green", line_width=2, name="trail")

        # Update limo
        if limo is not None:
            limo_pv = mqtt_to_pyvista(limo.pos)
            limo_mesh.copy_from(pv.Sphere(radius=LIMO_RADIUS, center=limo_pv))

        # Update HUD
        plotter.add_text(
            f"pos: [{drone.pos[0]:+.3f}, {drone.pos[1]:+.3f}, {drone.pos[2]:+.3f}]\n"
            f"vel: [{drone.vel[0]:+.3f}, {drone.vel[1]:+.3f}, {drone.vel[2]:+.3f}]\n"
            f"euler: [{drone.euler[0]:+.2f}, {drone.euler[1]:+.2f}, {drone.euler[2]:+.2f}]",
            position="upper_right", font_size=10, name="hud",
        )

    plotter.add_timer_event(
        max_steps=999_999, duration=CALLBACK_MS, callback=update_callback
    )

    print("\n=== MQTT Drone Viewer ===")
    print(f"Broker: {args.broker}:{args.port}")
    print(f"Topics: {DRONE_TOPIC}, {LIMO_TOPIC}")
    print("Close the window to exit.")
    print("=" * 26)

    plotter.show()
    client.loop_stop()
    client.disconnect()
    print("Done.")


if __name__ == "__main__":
    main()
