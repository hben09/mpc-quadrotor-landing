"""Reusable parser for MQTT rigid-body pose messages."""

import json
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class MQTTRigidBody:
    """Parsed rigid-body state from a single MQTT message."""

    pos: list[float]        # [x, y, z] metres
    rot: list[float]        # [qx, qy, qz, qw] quaternion
    euler: list[float]      # [roll, yaw, pitch] radians (intrinsic XYZ in OptiTrack frame)
                            # NOTE: euler[1] is gimbal-locked to ±π/2; use `yaw` for full range.
    yaw: float              # heading about Y/up, full ±π range, computed directly from quaternion
    vel: list[float]        # [vx, vy, vz] m/s (zero until tracker computes it)
    metadata: dict          # raw metadata dict
    timestamp: float        # motive_timestamp (seconds)


def parse_rigid_body(payload: str) -> MQTTRigidBody:
    """Parse a JSON payload string into an MQTTRigidBody.

    Expected JSON format::

        {
          "pos": [x, y, z],
          "rot": [qx, qy, qz, qw],
          "metadata": { "motive_timestamp": ..., ... }
        }
    """
    data = json.loads(payload)

    pos = data["pos"]
    rot = data["rot"]
    euler = Rotation.from_quat(rot).as_euler("xyz").tolist()
    # Yaw direct from quaternion — bypasses Euler decomposition so no gimbal lock at ±π/2.
    qx, qy, qz, qw = rot
    yaw = float(np.arctan2(2 * (qw * qy + qx * qz),
                           1 - 2 * (qy * qy + qz * qz)))
    metadata = data.get("metadata", {})
    timestamp = metadata.get("motive_timestamp", 0.0)

    return MQTTRigidBody(
        pos=pos,
        rot=rot,
        euler=euler,
        yaw=yaw,
        vel=[0.0, 0.0, 0.0],
        metadata=metadata,
        timestamp=timestamp,
    )


class RigidBodyTracker:
    """Track a rigid body over time and compute velocity via finite differencing."""

    def __init__(self):
        self._prev: MQTTRigidBody | None = None

    def update(self, payload: str) -> MQTTRigidBody:
        """Parse a new message and compute velocity from the previous one."""
        rb = parse_rigid_body(payload)

        if self._prev is not None:
            dt = rb.timestamp - self._prev.timestamp
            if dt > 0:
                rb.vel = [
                    (rb.pos[i] - self._prev.pos[i]) / dt for i in range(3)
                ]

        self._prev = rb
        return rb
