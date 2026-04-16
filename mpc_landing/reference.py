"""
Reference trajectory generation for MPC.

Builds the (N+1, 6) reference array that the MPC tracks.
Different functions for different phases / strategies.

No ROS, no sim dependencies — just numpy.
"""

import numpy as np


def tracking_reference(drone_state, limo_state, N, dt):
    """Phase 1: Track the Limo at a fixed altitude above it.

    Predicts Limo trajectory using constant velocity model.
    Drone reference = Limo xy position + fixed altitude offset.

    Args:
        drone_state: dict with 'pos' [x, y, z] and 'vel' [vx, vy, vz]
        limo_state:  dict with 'pos' [x, y, z] and 'vel' [vx, vy, vz]
        N: prediction horizon (number of steps)
        dt: timestep (s)

    Returns:
        ref: np.array of shape (N+1, 6) — [px, vx, py, vy, pz, vz] per step
    """
    ref = np.zeros((N + 1, 6))

    limo_pos = np.array(limo_state["pos"])
    limo_vel = np.array(limo_state["vel"])

    # Altitude to hold above the Limo
    tracking_altitude = limo_pos[1] + 0.5  # 0.5m above Limo

    for k in range(N + 1):
        t = k * dt
        pred_pos = limo_pos + limo_vel * t

        ref[k, 0] = pred_pos[0]  # px — match Limo x
        ref[k, 1] = limo_vel[0]  # vx — match Limo speed
        ref[k, 2] = tracking_altitude  # py — hold altitude above Limo
        ref[k, 3] = 0.0  # vy — no vertical speed while tracking
        ref[k, 4] = pred_pos[2]  # pz — match Limo z
        ref[k, 5] = limo_vel[2]  # vz — match Limo speed

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

    # Landing target = Limo roof height (small offset for clearance)
    landing_height = limo_pos[1] + 0.05

    for k in range(N + 1):
        t = k * dt
        pred_limo_pos = limo_pos + limo_vel * t

        # xy: track Limo
        ref[k, 0] = pred_limo_pos[0]
        ref[k, 1] = limo_vel[0]
        ref[k, 4] = pred_limo_pos[2]
        ref[k, 5] = limo_vel[2]

        # y (altitude): descend from current height toward Limo
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
