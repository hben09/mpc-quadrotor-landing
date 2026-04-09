"""
Arena boundary checker for RASTIC mocap arena.
Returns True if the drone is near any wall.
"""


# Arena limits (meters) — adjust to match RASTIC arena
ARENA_BOUNDS = {
    'x_min': -4.5, 'x_max': 3.0,
    'y_min':  0.0, 'y_max': 2.0,
    'z_min': -2.0, 'z_max': 3.0,
}

# How close to a wall before triggering (meters)
THRESHOLDS = {'x': 0.3, 'y': 0.2, 'z': 0.3}


def check_boundary(position):
    """Check if drone is within threshold of any arena wall.

    Args:
        position: [x, y, z] in meters.

    Returns:
        True if near any boundary (should return home).
    """
    x, y, z = position
    b = ARENA_BOUNDS
    t = THRESHOLDS

    if (b['x_max'] - x) <= t['x'] or (x - b['x_min']) <= t['x']:
        return True
    if (b['y_max'] - y) <= t['y']:
        return True
    if (b['z_max'] - z) <= t['z'] or (z - b['z_min']) <= t['z']:
        return True

    return False
