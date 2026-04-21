"""
Reference trajectory generation for MPC.

Builds the (N+1, 6) reference array that the MPC tracks.
Different functions for different phases / strategies.

No ROS, no sim dependencies — just numpy.
"""

import numpy as np

_CTRV_EPS = 1e-3  # rad/s — below this fall back to straight-line CV model


def _ctrv_predict(pos_x, pos_z, vx, vz, w, t):
    """Closed-form CTRV prediction in the OptiTrack xz plane (+Y up, CCW yaw).

    Integrates a velocity vector rotating at constant turn-rate w:
        pos(t) = pos_0 + int_0^t R_y(w*s) . v_0 ds
    """
    if abs(w) > _CTRV_EPS:
        sin_wt = np.sin(w * t)
        cos_wt = np.cos(w * t)
        dx = (sin_wt / w) * vx + ((1.0 - cos_wt) / w) * vz
        dz = (-(1.0 - cos_wt) / w) * vx + (sin_wt / w) * vz
        vx_t = vx * cos_wt + vz * sin_wt
        vz_t = -vx * sin_wt + vz * cos_wt
    else:
        dx, dz = vx * t, vz * t
        vx_t, vz_t = vx, vz
    return pos_x + dx, pos_z + dz, vx_t, vz_t


def tracking_reference(drone_state, limo_state, N, dt):
    """Phase 1: Track the Limo at a fixed altitude above it.

    Predicts Limo trajectory using a CTRV model (constant turn-rate and
    velocity) when `limo_state['yaw_rate']` is provided; degrades to pure
    constant-velocity (straight line) when yaw_rate is absent or near zero.
    Drone reference = predicted Limo xz position + fixed altitude offset.

    Args:
        drone_state: dict with 'pos' [x, y, z] and 'vel' [vx, vy, vz]
        limo_state:  dict with 'pos', 'vel', and optional 'yaw_rate' (rad/s)
        N: prediction horizon (number of steps)
        dt: timestep (s)

    Returns:
        ref: np.array of shape (N+1, 6) — [px, vx, py, vy, pz, vz] per step
    """
    ref = np.zeros((N + 1, 6))

    limo_pos = np.array(limo_state["pos"])
    limo_vel = np.array(limo_state["vel"])
    w = limo_state.get("yaw_rate", 0.0)

    tracking_altitude = limo_pos[1] + 0.5  # 0.5m above Limo

    for k in range(N + 1):
        t = k * dt
        px, pz, vx_t, vz_t = _ctrv_predict(
            limo_pos[0], limo_pos[2], limo_vel[0], limo_vel[2], w, t
        )
        ref[k, 0] = px
        ref[k, 1] = vx_t
        ref[k, 2] = tracking_altitude
        ref[k, 3] = 0.0
        ref[k, 4] = pz
        ref[k, 5] = vz_t

    return ref


def landing_reference(drone_state, limo_state, N, dt, descent_rate=0.3):
    """Phase 2: Descend onto the Limo while tracking its xy position.

    Same xy tracking as Phase 1, but altitude reference linearly descends
    from current drone height down to the Limo roof.

    Args:
        drone_state: dict with 'pos' [x, y, z] and 'vel' [vx, vy, vz]
        limo_state:  dict with 'pos' [x, y, z] and 'vel' [vx, vy, vz]
        N: prediction horizon
        dt: timestep (s)
        descent_rate: how fast to descend (m/s)

    Returns:
        ref: np.array of shape (N+1, 6)
    """
    ref = np.zeros((N + 1, 6))

    drone_pos = np.array(drone_state["pos"])
    limo_pos = np.array(limo_state["pos"])
    limo_vel = np.array(limo_state["vel"])
    w = limo_state.get("yaw_rate", 0.0)

    landing_height = limo_pos[1] + 0.05

    for k in range(N + 1):
        t = k * dt
        px, pz, vx_t, vz_t = _ctrv_predict(
            limo_pos[0], limo_pos[2], limo_vel[0], limo_vel[2], w, t
        )
        ref[k, 0] = px
        ref[k, 1] = vx_t
        ref[k, 4] = pz
        ref[k, 5] = vz_t

        desired_alt = drone_pos[1] - descent_rate * t
        ref[k, 2] = max(desired_alt, landing_height)
        ref[k, 3] = -descent_rate if desired_alt > landing_height else 0.0

    return ref


def static_reference(target_pos, N, dt):
    """Go-to-point reference (for testing or return-home).

    Args:
        target_pos: [x, y, z] target position
        N: prediction horizon
        dt: timestep (s)

    Returns:
        ref: np.array of shape (N+1, 6)
    """
    ref = np.zeros((N + 1, 6))
    for k in range(N + 1):
        ref[k, 0] = target_pos[0]  # px
        ref[k, 2] = target_pos[1]  # py
        ref[k, 4] = target_pos[2]  # pz
        # velocities stay 0 — we want to stop at the target

    return ref
