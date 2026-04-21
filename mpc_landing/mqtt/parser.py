"""Reusable parser for MQTT rigid-body pose messages.

`RigidBodyTracker` runs a constant-acceleration Kalman filter per axis on
the position stream and fills the returned `MQTTRigidBody.vel` (and `.accel`)
with the filtered estimates. This replaces the previous finite-difference
velocity, which was one sample stale and very noisy — both showing up as the
drone "tracking behind" a moving target during landings.

The filter is per-axis and identical across x/y/z. Each axis has state
[p, v, a] and observes p directly. Process noise is driven by a jerk-noise
model (σ_j · dt^k / k!) so a single `sigma_jerk` parameter tunes how agile
the filter is.

This module is self-contained (numpy only) and unit-testable without any
hardware — see tests/test_kalman_tracker.py.
"""

import json
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation


# Default tuning — good starting point for OptiTrack at ~120 Hz.
# sigma_jerk: how much instantaneous acceleration change we expect (m/s³).
#   LIMO doing a figure-8 has peak jerk of O(1) m/s³; 2.0 gives headroom.
# sigma_meas: OptiTrack single-marker positional noise (m). 3 mm is typical.
DEFAULT_SIGMA_JERK = 2.0     # m/s³
DEFAULT_SIGMA_MEAS = 0.003   # m


@dataclass
class MQTTRigidBody:
    """Parsed rigid-body state from a single MQTT message."""

    pos: list[float]        # [x, y, z] metres
    rot: list[float]        # [qx, qy, qz, qw] quaternion
    euler: list[float]      # [roll, yaw, pitch] radians (intrinsic XYZ in OptiTrack frame)
                            # NOTE: euler[1] is gimbal-locked to ±π/2; use `yaw` for full range.
    yaw: float              # heading about Y/up, full ±π range, computed directly from quaternion
    vel: list[float]        # [vx, vy, vz] m/s — Kalman-filtered (zero before first update)
    accel: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
                            # [ax, ay, az] m/s² — Kalman-filtered
    metadata: dict = field(default_factory=dict)
    timestamp: float = 0.0  # motive_timestamp (seconds)


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


class _AxisKalman:
    """Single-axis constant-acceleration Kalman filter.

    State x = [position, velocity, acceleration]^T.
    Measurement z = position.

    Transition over dt:
        x_k+1 = F(dt) · x_k + w,   with w ~ N(0, Q(dt))
    where F(dt) = [[1, dt, dt^2/2], [0, 1, dt], [0, 0, 1]]
    and Q(dt) is the discrete-time covariance for continuous jerk noise
    with intensity sigma_jerk² (standard CA model, Bar-Shalom eq. 6.3.2-4).
    """

    def __init__(self, sigma_jerk: float, sigma_meas: float):
        self.sigma_jerk = sigma_jerk
        self.sigma_meas = sigma_meas
        self.x = np.zeros(3)           # [p, v, a]
        self.P = np.eye(3) * 1e3       # large initial uncertainty
        self._initialized = False

    def initialize(self, p0: float) -> None:
        self.x[0] = p0
        self.x[1] = 0.0
        self.x[2] = 0.0
        self.P = np.diag([self.sigma_meas ** 2, 1.0, 10.0])
        self._initialized = True

    def predict(self, dt: float) -> None:
        F = np.array([
            [1.0, dt, 0.5 * dt * dt],
            [0.0, 1.0, dt],
            [0.0, 0.0, 1.0],
        ])
        q = self.sigma_jerk ** 2
        # Discrete white-jerk process noise (canonical CA form).
        dt2, dt3, dt4, dt5 = dt ** 2, dt ** 3, dt ** 4, dt ** 5
        Q = q * np.array([
            [dt5 / 20.0, dt4 / 8.0,  dt3 / 6.0],
            [dt4 / 8.0,  dt3 / 3.0,  dt2 / 2.0],
            [dt3 / 6.0,  dt2 / 2.0,  dt],
        ])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z: float) -> None:
        # H = [1, 0, 0]; innovation is scalar.
        H = np.array([1.0, 0.0, 0.0])
        R = self.sigma_meas ** 2
        y = z - H @ self.x
        S = H @ self.P @ H + R          # scalar
        K = self.P @ H / S              # 3-vector
        self.x = self.x + K * y
        self.P = (np.eye(3) - np.outer(K, H)) @ self.P

    @property
    def pos(self) -> float:
        return float(self.x[0])

    @property
    def vel(self) -> float:
        return float(self.x[1])

    @property
    def accel(self) -> float:
        return float(self.x[2])


class RigidBodyTracker:
    """Track a rigid body and smooth position/velocity/acceleration via Kalman.

    Drop-in replacement for the previous finite-difference tracker: same
    `update(payload) -> MQTTRigidBody` API, with `.vel` now filtered and a
    new `.accel` field available.

    Use per-instance (one tracker per rigid body) so the filter state
    doesn't cross between, e.g., the drone and the landing target.
    """

    def __init__(
        self,
        sigma_jerk: float = DEFAULT_SIGMA_JERK,
        sigma_meas: float = DEFAULT_SIGMA_MEAS,
    ):
        self._filters = [_AxisKalman(sigma_jerk, sigma_meas) for _ in range(3)]
        self._last_timestamp: float | None = None

    def update(self, payload: str) -> MQTTRigidBody:
        rb = parse_rigid_body(payload)

        if self._last_timestamp is None:
            # First sample: seed each axis with the measurement.
            for i in range(3):
                self._filters[i].initialize(rb.pos[i])
        else:
            dt = rb.timestamp - self._last_timestamp
            if dt > 0:
                for i in range(3):
                    self._filters[i].predict(dt)
                    self._filters[i].update(rb.pos[i])
            # If dt <= 0 (duplicate or out-of-order timestamp), skip this sample.

        self._last_timestamp = rb.timestamp

        rb.vel = [f.vel for f in self._filters]
        rb.accel = [f.accel for f in self._filters]
        return rb
