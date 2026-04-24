"""Flight-log plotter for MPC report figures.

Reads one logs/teleop_YYYYMMDD_HHMMSS/ directory and emits vector
PDFs into a ``plots/`` sub-directory, plus a summary.txt with headline
numbers.

Usage:
    uv run plot-flight                    # most recent flight
    uv run plot-flight <log-dir>          # specific flight
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"

_EVENT_KINDS = {
    "takeoff_complete",
    "mode_change",
    "touchdown",
    "motors_off",
    "lost_pose",
    "boundary_violated",
}


def resolve_log_dir(arg: str | None) -> Path:
    if arg:
        p = Path(arg).expanduser().resolve()
        if not p.is_dir():
            sys.exit(f"error: log directory not found: {p}")
        return p
    candidates = sorted(LOGS_DIR.glob("teleop_*"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        sys.exit(f"error: no teleop_* directories under {LOGS_DIR}")
    return candidates[-1]


def load_csv(log_dir: Path) -> dict[str, np.ndarray]:
    path = log_dir / "teleop.csv"
    if not path.is_file():
        sys.exit(f"error: missing teleop.csv at {path}")
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    if not rows:
        sys.exit(f"error: empty teleop.csv at {path}")

    cols: dict[str, list[str]] = {k: [] for k in fieldnames}
    for row in rows:
        for k in fieldnames:
            cols[k].append(row[k])

    out: dict[str, np.ndarray] = {}
    for k, vals in cols.items():
        try:
            out[k] = np.array(
                [float(v) if v != "" else math.nan for v in vals],
                dtype=float,
            )
        except ValueError:
            out[k] = np.array(vals, dtype=object)
    return out


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    items: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _draw_event_lines(ax, events: list[dict]) -> None:
    for ev in events:
        if ev.get("kind") not in _EVENT_KINDS:
            continue
        ax.axvline(ev["t"], color="0.4", linestyle="--", linewidth=0.8, alpha=0.7)


def _annotate_event_labels(ax, events: list[dict]) -> None:
    """Place rotated event labels above the given axes (axes-fraction y)."""
    trans = ax.get_xaxis_transform()
    for ev in events:
        kind = ev.get("kind")
        if kind not in _EVENT_KINDS:
            continue
        label = (
            f"{ev.get('from_mode', '?')}→{ev.get('to_mode', '?')}"
            if kind == "mode_change"
            else kind
        )
        ax.text(
            ev["t"],
            1.02,
            label,
            transform=trans,
            fontsize=7,
            color="0.3",
            rotation=90,
            va="bottom",
            ha="center",
        )


def _style(ax) -> None:
    ax.grid(True, linestyle=":", alpha=0.5)


def _save(fig, path: Path) -> None:
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_position_tracking(csv_data: dict, events: list, out: Path) -> None:
    t = csv_data["t"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 6))

    pos_panels = [
        ("x-Position (m)", "tx", "px", "pad_x"),
        ("y-Position (m)", "ty", "py", "pad_y"),
        ("z-Position (m)", "tz", "pz", "pad_z"),
    ]
    vel_panels = [
        ("x-Velocity (m/s)", "tvx", "vx", "pad_vx"),
        ("y-Velocity (m/s)", "tvy", "vy", "pad_vy"),
        ("z-Velocity (m/s)", "tvz", "vz", "pad_vz"),
    ]

    def _draw_row(row_axes, panels):
        for ax, (ylab, desired_col, actual_col, pad_col) in zip(row_axes, panels):
            ax.plot(
                t,
                csv_data[desired_col],
                color="C0",
                linewidth=1.3,
                label="Quadrotor Desired",
            )
            ax.plot(
                t,
                csv_data[actual_col],
                color="C1",
                linewidth=1.3,
                label="Quadrotor Ground Truth",
            )
            ax.plot(
                t,
                csv_data[pad_col],
                color="C4",
                linewidth=1.3,
                label="Platform Ground Truth",
            )
            ax.set_ylabel(ylab)
            ax.set_xlabel("Time (s)")
            _draw_event_lines(ax, events)
            _style(ax)

    _draw_row(axes[0], pos_panels)
    _draw_row(axes[1], vel_panels)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, 1.0),
        frameon=False,
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, out / "01_position_tracking.pdf")


def plot_control_output(csv_data: dict, events: list, out: Path) -> None:
    t = csv_data["t"]
    a_max = (
        float(np.nanmax(csv_data["a_max"]))
        if "a_max" in csv_data and not np.all(np.isnan(csv_data["a_max"]))
        else None
    )
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))

    accel_panels = [
        ("x-Acceleration (m/s²)", "ax", (-5, 5)),
        ("y-Acceleration (m/s²)", "ay", (0, 15)),
        ("z-Acceleration (m/s²)", "az", (-5, 5)),
    ]
    for ax, (ylab, col, ylim) in zip(axes[0], accel_panels):
        ax.plot(t, csv_data[col], color="C3", linewidth=1.3, label="MPC Command")
        if a_max is not None:
            ax.axhline(
                a_max,
                color="0.5",
                linestyle=":",
                linewidth=0.8,
                label=f"Acceleration Limit (±{a_max:.1f} m/s²)",
            )
            ax.axhline(-a_max, color="0.5", linestyle=":", linewidth=0.8)
        ax.set_ylabel(ylab)
        ax.set_xlabel("Time (s)")
        ax.set_ylim(ylim)
        _draw_event_lines(ax, events)
        _style(ax)

    actuator_panels = [
        ("Roll Command (deg)", "roll", 15, "Tilt Limit (±15°)", True),
        ("Pitch Command (deg)", "pitch", 15, None, True),
        ("Thrust Command (PWM)", "thrust", 65535, "PWM Maximum (65535)", False),
    ]
    for ax, (ylab, col, limit_val, limit_label, symmetric) in zip(
        axes[1], actuator_panels
    ):
        ax.plot(t, csv_data[col], color="C0", linewidth=1.3, label="Actuator Command")
        if limit_label is not None:
            ax.axhline(
                limit_val, color="0.5", linestyle=":", linewidth=0.8, label=limit_label
            )
        else:
            ax.axhline(limit_val, color="0.5", linestyle=":", linewidth=0.8)
        if symmetric:
            ax.axhline(-limit_val, color="0.5", linestyle=":", linewidth=0.8)
        ax.set_ylabel(ylab)
        ax.set_xlabel("Time (s)")
        _draw_event_lines(ax, events)
        _style(ax)

    handles_labels: dict[str, object] = {}
    for ax in axes.flat:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles_labels.setdefault(label, handle)
    fig.legend(
        handles_labels.values(),
        handles_labels.keys(),
        loc="upper center",
        ncol=len(handles_labels),
        bbox_to_anchor=(0.5, 1.0),
        frameon=False,
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, out / "02_commands.pdf")


def plot_disturbance(csv_data: dict, events: list, out: Path) -> None:
    t = csv_data["t"]
    fig, ax = plt.subplots(1, 1, figsize=(8, 3.6))
    ax.plot(t, csv_data["d_hat"], color="C4", linewidth=1.3)
    ax.set_ylabel("Vertical disturbance estimate (m/s²)")
    ax.set_xlabel("Time since takeoff (s)")
    ax.set_title("Offset-free MPC vertical disturbance estimate")
    _draw_event_lines(ax, events)
    _style(ax)
    _save(fig, out / "03_disturbance_estimate.pdf")


def _rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean(a * a)))


def _rms_annotation(ax, value: float, unit: str) -> None:
    ax.text(
        0.98,
        0.95,
        f"RMS = {value:.3f} {unit}",
        transform=ax.transAxes,
        fontsize=9,
        ha="right",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 2.5},
    )


def plot_tracking_error(csv_data: dict, events: list, out: Path) -> None:
    t = csv_data["t"]
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.25)

    pos_panels = [
        ("x-Position Error (m)", "e_px"),
        ("y-Position Error (m)", "e_py"),
        ("z-Position Error (m)", "e_pz"),
    ]
    for i, (ylab, col) in enumerate(pos_panels):
        ax = fig.add_subplot(gs[0, i])
        ax.plot(t, csv_data[col], color="C0", linewidth=1.3)
        ax.set_ylabel(ylab)
        ax.set_xlabel("Time (s)")
        _rms_annotation(ax, _rms(csv_data[col]), "m")
        _draw_event_lines(ax, events)
        _style(ax)

    vel_panels = [
        ("x-Velocity Error (m/s)", "e_vx"),
        ("y-Velocity Error (m/s)", "e_vy"),
        ("z-Velocity Error (m/s)", "e_vz"),
    ]
    for i, (ylab, col) in enumerate(vel_panels):
        ax = fig.add_subplot(gs[1, i])
        ax.plot(t, csv_data[col], color="C0", linewidth=1.3)
        ax.set_ylabel(ylab)
        ax.set_xlabel("Time (s)")
        _rms_annotation(ax, _rms(csv_data[col]), "m/s")
        _draw_event_lines(ax, events)
        _style(ax)

    ax = fig.add_subplot(gs[2, :])
    e_yaw_deg = np.degrees(csv_data["e_yaw"])
    ax.plot(t, e_yaw_deg, color="C3", linewidth=1.3)
    ax.set_ylabel("Yaw Error (deg)")
    ax.set_xlabel("Time (s)")
    _rms_annotation(ax, _rms(e_yaw_deg), "deg")
    _draw_event_lines(ax, events)
    _style(ax)

    _save(fig, out / "04_tracking_error.pdf")


def write_summary(csv_data: dict, events: list, infeasible: list, out: Path) -> None:
    t = csv_data["t"]
    duration = float(t[-1] - t[0])
    solve_ms = csv_data["solve_time_ms"]
    valid = solve_ms[~np.isnan(solve_ms)]
    p50 = float(np.percentile(valid, 50)) if valid.size else float("nan")
    p95 = float(np.percentile(valid, 95)) if valid.size else float("nan")
    p99 = float(np.percentile(valid, 99)) if valid.size else float("nan")
    touchdown = next((ev for ev in events if ev.get("kind") == "touchdown"), None)

    lines = [
        f"Flight duration: {duration:.2f} s  ({len(t)} samples)",
        f"Position RMS error: x={_rms(csv_data['e_px']):.3f} m, "
        f"y={_rms(csv_data['e_py']):.3f} m, z={_rms(csv_data['e_pz']):.3f} m",
        f"Velocity RMS error: vx={_rms(csv_data['e_vx']):.3f} m/s, "
        f"vy={_rms(csv_data['e_vy']):.3f} m/s, vz={_rms(csv_data['e_vz']):.3f} m/s",
        f"Yaw RMS error: {_rms(np.degrees(csv_data['e_yaw'])):.2f}°",
        f"Solve time: p50={p50:.2f} ms, p95={p95:.2f} ms, p99={p99:.2f} ms",
        f"Infeasible solves: {len(infeasible)}",
        f"Events: {len(events)}",
    ]
    if touchdown:
        lines.append(
            f"Touchdown: y={touchdown.get('y'):.3f} m, "
            f"pad_y={touchdown.get('pad_y'):.3f} m, "
            f"ramp={touchdown.get('ramp_duration_s')} s"
        )
    with (out / "summary.txt").open("w") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot an MPC flight log for a report.")
    ap.add_argument(
        "log_dir",
        nargs="?",
        help="path to a logs/teleop_* directory (default: most recent)",
    )
    args = ap.parse_args()

    log_dir = resolve_log_dir(args.log_dir)
    print(f"Plotting flight log: {log_dir}")

    csv_data = load_csv(log_dir)
    events = load_jsonl(log_dir / "events.jsonl")
    infeasible = load_jsonl(log_dir / "infeasible.jsonl")

    plots_dir = log_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    plot_position_tracking(csv_data, events, plots_dir)
    plot_control_output(csv_data, events, plots_dir)
    plot_disturbance(csv_data, events, plots_dir)
    plot_tracking_error(csv_data, events, plots_dir)
    write_summary(csv_data, events, infeasible, plots_dir)

    print(f"Wrote 4 PDFs and summary.txt to {plots_dir}")


if __name__ == "__main__":
    main()
