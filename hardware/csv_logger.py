"""CSV logging helper for hardware MPC teleop scripts."""

import csv
from datetime import datetime
from pathlib import Path

_COLUMNS = [
    "t",
    "px",
    "py",
    "pz",
    "vx",
    "vy",
    "vz",
    "tx",
    "ty",
    "tz",
    "ax",
    "ay",
    "az",
    "roll",
    "pitch",
    "thrust",
    "d_hat",
    "yaw",
    "target_yaw",
    "yawrate",
    "Qp",
    "Qv",
    "Qf",
    "R",
    "a_max",
    "v_max",
]


class TeleopLogger:
    """Append-only CSV logger matching the shared MPC teleop schema.

    Pass ``include_mode=True`` to append a trailing ``mode`` column
    (used by mpc_teleop_landing.py to record "T"/"M").
    """

    def __init__(self, log_dir: Path, *, include_mode: bool = False):
        log_dir.mkdir(exist_ok=True)
        self.path = log_dir / f"teleop_{datetime.now():%Y%m%d_%H%M%S}.csv"
        self._include_mode = include_mode
        self._file = None
        self._writer = None

    def __enter__(self):
        self._file = open(self.path, "w", newline="")
        self._writer = csv.writer(self._file)
        header = list(_COLUMNS) + (["mode"] if self._include_mode else [])
        self._writer.writerow(header)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._file is not None:
            self._file.close()

    def log(
        self,
        *,
        t,
        pos,
        vel,
        target,
        accel,
        setpoint,
        disturbance,
        yaw,
        target_yaw,
        yawrate,
        config,
        mode=None,
    ):
        roll, pitch, thrust = setpoint
        row = [
            f"{t:.3f}",
            f"{pos[0]:.4f}",
            f"{pos[1]:.4f}",
            f"{pos[2]:.4f}",
            f"{vel[0]:.4f}",
            f"{vel[1]:.4f}",
            f"{vel[2]:.4f}",
            f"{target[0]:.4f}",
            f"{target[1]:.4f}",
            f"{target[2]:.4f}",
            f"{accel[0]:.4f}",
            f"{accel[1]:.4f}",
            f"{accel[2]:.4f}",
            f"{roll:.2f}",
            f"{pitch:.2f}",
            f"{thrust}",
            f"{disturbance:.4f}",
            f"{yaw:.4f}",
            f"{target_yaw:.4f}",
            f"{yawrate:.2f}",
            f"{config.Q_diag[0]:.2f}",
            f"{config.Q_diag[1]:.2f}",
            f"{config.Qf_diag[0]:.2f}",
            f"{config.R_diag[0]:.2f}",
            f"{config.a_max:.2f}",
            f"{config.v_max:.2f}",
        ]
        if self._include_mode:
            row.append(mode if mode is not None else "")
        self._writer.writerow(row)
