"""CSV logging helper for hardware MPC teleop scripts.

Each flight's artifacts (teleop.csv, events.jsonl, infeasible.jsonl) live
together in a per-flight directory chosen by the caller. The timestamp
belongs to the directory name; filenames stay short.
"""

import csv
import json
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
    "mpc_status",
    "solve_time_ms",
    "loop_dt_ms",
    "a_cmd_norm",
    "vbat",
    "e_px",
    "e_py",
    "e_pz",
    "e_vx",
    "e_vy",
    "e_vz",
    "e_yaw",
    "tvx",
    "tvy",
    "tvz",
    "pad_x",
    "pad_y",
    "pad_z",
    "pad_vx",
    "pad_vy",
    "pad_vz",
]


class TeleopLogger:
    """Append-only CSV logger matching the shared MPC teleop schema.

    Pass ``include_mode=True`` to append a trailing ``mode`` column
    (used by mpc_pilot.py to record "T"/"M").
    """

    def __init__(self, log_dir: Path, *, include_mode: bool = False):
        log_dir.mkdir(parents=True, exist_ok=True)
        self.path = log_dir / "teleop.csv"
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
        mpc_status="",
        mode=None,
        solve_time_ms=None,
        loop_dt_ms=None,
        a_cmd_norm=None,
        vbat=None,
        pos_error,
        vel_error,
        yaw_error,
        target_vel,
        pad_pos=None,
        pad_vel=None,
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
            f"{mpc_status}",
            "" if solve_time_ms is None else f"{solve_time_ms:.3f}",
            "" if loop_dt_ms is None else f"{loop_dt_ms:.3f}",
            "" if a_cmd_norm is None else f"{a_cmd_norm:.4f}",
            "" if vbat is None else f"{vbat:.3f}",
            f"{pos_error[0]:.4f}",
            f"{pos_error[1]:.4f}",
            f"{pos_error[2]:.4f}",
            f"{vel_error[0]:.4f}",
            f"{vel_error[1]:.4f}",
            f"{vel_error[2]:.4f}",
            f"{yaw_error:.4f}",
            f"{target_vel[0]:.4f}",
            f"{target_vel[1]:.4f}",
            f"{target_vel[2]:.4f}",
            "" if pad_pos is None else f"{pad_pos[0]:.4f}",
            "" if pad_pos is None else f"{pad_pos[1]:.4f}",
            "" if pad_pos is None else f"{pad_pos[2]:.4f}",
            "" if pad_vel is None else f"{pad_vel[0]:.4f}",
            "" if pad_vel is None else f"{pad_vel[1]:.4f}",
            "" if pad_vel is None else f"{pad_vel[2]:.4f}",
        ]
        if self._include_mode:
            row.append(mode if mode is not None else "")
        self._writer.writerow(row)


class InfeasibilityLogger:
    def __init__(self, log_dir: Path):
        log_dir.mkdir(parents=True, exist_ok=True)
        self.path = log_dir / "infeasible.jsonl"
        self._file = None

    def __enter__(self):
        self._file = open(self.path, "w")
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._file is not None:
            self._file.close()

    def log(self, *, t, x0, ref, d_hat, config, status):
        ev = {
            "t": float(t),
            "status": str(status),
            "d_hat": float(d_hat),
            "x0": [float(v) for v in x0],
            "ref": ref.tolist(),
            "config": {
                "dt": config.dt,
                "horizon": config.horizon,
                "Q_diag": list(config.Q_diag),
                "Qf_diag": list(config.Qf_diag),
                "R_diag": list(config.R_diag),
                "a_max": config.a_max,
                "v_max": config.v_max,
            },
        }
        self._file.write(json.dumps(ev) + "\n")
        self._file.flush()


class EventLogger:
    """JSONL log for discrete flight events (mode toggles, touchdown, etc.)."""

    def __init__(self, log_dir: Path):
        log_dir.mkdir(parents=True, exist_ok=True)
        self.path = log_dir / "events.jsonl"
        self._file = None

    def __enter__(self):
        self._file = open(self.path, "w")
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._file is not None:
            self._file.close()

    def log(self, t, kind, **fields):
        ev = {"t": float(t), "kind": str(kind), **fields}
        self._file.write(json.dumps(ev) + "\n")
        self._file.flush()
