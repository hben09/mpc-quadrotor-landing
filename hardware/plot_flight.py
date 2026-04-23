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
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 7.5))
    panels = [
        ("x — forward (m)", "px", "tx", "pad_x"),
        ("y — up (m)", "py", "ty", "pad_y"),
        ("z — right (m)", "pz", "tz", "pad_z"),
    ]
    for ax, (ylab, pcol, tcol, pad_col) in zip(axes, panels):
        ax.plot(t, csv_data[pcol], label="drone (actual)", color="C0", linewidth=1.3)
        ax.plot(
            t,
            csv_data[tcol],
            label="MPC reference",
            color="C1",
            linewidth=1.2,
            linestyle="--",
        )
        ax.plot(
            t,
            csv_data[pad_col],
            label="landing pad",
            color="C2",
            linewidth=1.0,
            alpha=0.9,
        )
        ax.set_ylabel(ylab)
        _draw_event_lines(ax, events)
        _style(ax)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time since takeoff start (s)")
    axes[0].set_title(
        "Position tracking — drone vs MPC reference vs landing pad",
        pad=24,
    )
    _annotate_event_labels(axes[0], events)
    _save(fig, out / "01_position_tracking.pdf")


def plot_control_output(csv_data: dict, events: list, out: Path) -> None:
    t = csv_data["t"]
    a_max = (
        float(np.nanmax(csv_data["a_max"]))
        if "a_max" in csv_data and not np.all(np.isnan(csv_data["a_max"]))
        else None
    )
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 6.5))
    panels = [("ax (m/s²)", "ax"), ("ay (m/s²)", "ay"), ("az (m/s²)", "az")]
    for ax, (ylab, col) in zip(axes, panels):
        ax.plot(t, csv_data[col], color="C3", linewidth=1.0)
        if a_max is not None:
            ax.axhline(
                a_max,
                color="0.5",
                linestyle=":",
                linewidth=0.8,
                label=f"±a_max = {a_max:.1f}",
            )
            ax.axhline(-a_max, color="0.5", linestyle=":", linewidth=0.8)
        ax.set_ylabel(ylab)
        _draw_event_lines(ax, events)
        _style(ax)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time since takeoff start (s)")
    axes[0].set_title("MPC commanded acceleration  u_opt[0]")
    _save(fig, out / "02_mpc_control_output.pdf")


def plot_disturbance(csv_data: dict, events: list, out: Path) -> None:
    t = csv_data["t"]
    fig, ax = plt.subplots(1, 1, figsize=(8, 3.6))
    ax.plot(t, csv_data["d_hat"], color="C4", linewidth=1.3)
    ax.set_ylabel("d_hat  (m/s²)")
    ax.set_xlabel("time since takeoff start (s)")
    ax.set_title("Offset-free MPC vertical disturbance estimate")
    _draw_event_lines(ax, events)
    _style(ax)
    _save(fig, out / "03_disturbance_estimate.pdf")


def plot_predicted_horizon(infeasible: list, out: Path) -> None:
    path = out / "04_predicted_horizon.pdf"
    if not infeasible:
        fig, ax = plt.subplots(1, 1, figsize=(8, 3.8))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No infeasible MPC solves recorded in this flight.\n\n"
            "The full 50-step predicted horizon is only logged to\n"
            "infeasible.jsonl on non-optimal solves. Extend the logger\n"
            "to snapshot horizons periodically if you need this figure\n"
            "on every flight.",
            ha="center",
            va="center",
            fontsize=10,
            family="monospace",
        )
        _save(fig, path)
        return

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 7.5))
    panels = [("ref x (m)", 0), ("ref y (m)", 2), ("ref z (m)", 4)]
    n = len(infeasible)
    for i, ev in enumerate(infeasible):
        ref = np.asarray(ev["ref"], dtype=float)
        x0 = np.asarray(ev["x0"], dtype=float)
        steps = np.arange(ref.shape[0])
        color = plt.cm.viridis(i / max(1, n - 1))
        label = f"t={ev['t']:.2f}s  ({ev.get('status', '?')})"
        for ax, (_, idx) in zip(axes, panels):
            ax.plot(steps, ref[:, idx], color=color, linewidth=1.0,
                    label=label if ax is axes[0] else None)
            ax.plot([0], [x0[idx]], marker="o", color=color, markersize=4)
    for ax, (ylab, _) in zip(axes, panels):
        ax.set_ylabel(ylab)
        _style(ax)
    axes[-1].set_xlabel("horizon step (20 ms each)")
    axes[0].set_title(f"MPC predicted horizons at non-optimal solves  (n={n})")
    axes[0].legend(loc="best", fontsize=7)
    _save(fig, path)


def _rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean(a * a)))


def plot_tracking_error(csv_data: dict, events: list, out: Path) -> None:
    t = csv_data["t"]
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 7.5))

    ax = axes[0]
    for col, color in [("e_px", "C0"), ("e_py", "C1"), ("e_pz", "C2")]:
        ax.plot(t, csv_data[col], color=color, linewidth=1.0, label=col)
    ax.set_ylabel("position error (m)")
    ax.set_title(
        "Position error  —  RMS "
        f"x={_rms(csv_data['e_px']):.3f}  "
        f"y={_rms(csv_data['e_py']):.3f}  "
        f"z={_rms(csv_data['e_pz']):.3f}"
    )
    ax.legend(loc="upper right", fontsize=8)
    _draw_event_lines(ax, events)
    _style(ax)

    ax = axes[1]
    for col, color in [("e_vx", "C0"), ("e_vy", "C1"), ("e_vz", "C2")]:
        ax.plot(t, csv_data[col], color=color, linewidth=1.0, label=col)
    ax.set_ylabel("velocity error (m/s)")
    ax.set_title(
        "Velocity error  —  RMS "
        f"vx={_rms(csv_data['e_vx']):.3f}  "
        f"vy={_rms(csv_data['e_vy']):.3f}  "
        f"vz={_rms(csv_data['e_vz']):.3f}"
    )
    ax.legend(loc="upper right", fontsize=8)
    _draw_event_lines(ax, events)
    _style(ax)

    ax = axes[2]
    e_yaw_deg = np.degrees(csv_data["e_yaw"])
    ax.plot(t, e_yaw_deg, color="C3", linewidth=1.0)
    ax.set_ylabel("yaw error (deg)")
    ax.set_title(f"Yaw error  —  RMS = {_rms(e_yaw_deg):.2f}°")
    _draw_event_lines(ax, events)
    _style(ax)

    axes[-1].set_xlabel("time since takeoff start (s)")
    _save(fig, out / "05_tracking_error.pdf")


def plot_solver_diagnostics(csv_data: dict, out: Path) -> None:
    t = csv_data["t"]
    solve_ms = csv_data["solve_time_ms"]
    status = list(csv_data["mpc_status"])

    counts: dict[str, int] = {}
    for s in status:
        counts[str(s)] = counts.get(str(s), 0) + 1
    title_counts = ", ".join(
        f"{k}: {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [2, 1]})

    ax = axes[0]
    ax.plot(t, solve_ms, color="C5", linewidth=0.9, alpha=0.9)
    valid = solve_ms[~np.isnan(solve_ms)]
    p50 = float(np.percentile(valid, 50)) if valid.size else float("nan")
    p95 = float(np.percentile(valid, 95)) if valid.size else float("nan")
    p99 = float(np.percentile(valid, 99)) if valid.size else float("nan")
    ax.axhline(p50, color="0.3", linestyle="--", linewidth=0.8, label=f"p50 = {p50:.1f} ms")
    ax.axhline(p95, color="0.5", linestyle="--", linewidth=0.8, label=f"p95 = {p95:.1f} ms")
    ax.axhline(20.0, color="red", linestyle=":", linewidth=0.8, label="dt = 20 ms")
    bad_mask = np.array(
        [s not in ("optimal", "optimal_inaccurate") for s in status],
        dtype=bool,
    )
    if bad_mask.any():
        ax.scatter(
            t[bad_mask],
            solve_ms[bad_mask],
            color="red",
            s=15,
            zorder=5,
            label=f"non-optimal  (n={int(bad_mask.sum())})",
        )
    ax.set_xlabel("time since takeoff start (s)")
    ax.set_ylabel("solve time (ms)")
    ax.legend(loc="upper right", fontsize=7)
    _style(ax)

    ax = axes[1]
    if valid.size:
        ax.hist(valid, bins=40, color="C5", alpha=0.8)
    ax.axvline(p50, color="0.3", linestyle="--", linewidth=0.8)
    ax.axvline(p95, color="0.5", linestyle="--", linewidth=0.8)
    ax.axvline(20.0, color="red", linestyle=":", linewidth=0.8)
    ax.set_xlabel("solve time (ms)")
    ax.set_ylabel("count")
    _style(ax)

    fig.suptitle(f"MPC solver diagnostics  —  {title_counts}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, out / "06_solver_diagnostics.pdf")


def plot_actuator_commands(csv_data: dict, events: list, out: Path) -> None:
    t = csv_data["t"]
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 7))

    axes[0].plot(t, csv_data["roll"], color="C0", linewidth=1.0)
    axes[0].axhline(15, color="0.5", linestyle=":", linewidth=0.8, label="±15° (hw-teleop limit)")
    axes[0].axhline(-15, color="0.5", linestyle=":", linewidth=0.8)
    axes[0].set_ylabel("roll cmd (deg)")
    axes[0].legend(loc="upper right", fontsize=7)

    axes[1].plot(t, csv_data["pitch"], color="C1", linewidth=1.0)
    axes[1].axhline(15, color="0.5", linestyle=":", linewidth=0.8)
    axes[1].axhline(-15, color="0.5", linestyle=":", linewidth=0.8)
    axes[1].set_ylabel("pitch cmd (deg)")

    axes[2].plot(t, csv_data["thrust"], color="C2", linewidth=1.0)
    axes[2].axhline(65535, color="0.5", linestyle=":", linewidth=0.8, label="PWM max 65535")
    axes[2].set_ylabel("thrust cmd (PWM)")
    axes[2].legend(loc="upper right", fontsize=7)

    for ax in axes:
        _draw_event_lines(ax, events)
        _style(ax)

    axes[-1].set_xlabel("time since takeoff start (s)")
    axes[0].set_title("Actuator commands sent to cflib")
    _save(fig, out / "07_actuator_commands.pdf")


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
    plot_predicted_horizon(infeasible, plots_dir)
    plot_tracking_error(csv_data, events, plots_dir)
    plot_solver_diagnostics(csv_data, plots_dir)
    plot_actuator_commands(csv_data, events, plots_dir)
    write_summary(csv_data, events, infeasible, plots_dir)

    print(f"Wrote 7 PDFs and summary.txt to {plots_dir}")


if __name__ == "__main__":
    main()
