"""MPC-based autonomous quadrotor landing system."""

from mpc_landing.mpc import MPCController, MPCConfig
from mpc_landing.reference import tracking_reference, landing_reference, static_reference
from mpc_landing.boundary import check_boundary, ARENA_BOUNDS
from mpc_landing.supervisor import SafeCommander
from mpc_landing.yaw_controller import compute_yawrate, wrap_to_pi

__all__ = [
    "MPCController",
    "MPCConfig",
    "tracking_reference",
    "landing_reference",
    "static_reference",
    "check_boundary",
    "ARENA_BOUNDS",
    "SafeCommander",
    "compute_yawrate",
    "wrap_to_pi",
]
