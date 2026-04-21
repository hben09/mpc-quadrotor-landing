"""Tests for the Kalman-filtered RigidBodyTracker.

Run with: `uv run python -m mpc_landing.mqtt.test_parser`
(or just `uv run python mpc_landing/mqtt/test_parser.py`).

We compare the new Kalman tracker against the old finite-difference behavior
on a realistic synthetic trajectory: a LIMO-scale figure-8 at 0.5 m/s with
3 mm mocap noise added. The Kalman filter should beat finite-diff on
velocity RMS error by a large margin, and should have smaller lag (measured
as cross-correlation peak).
"""

import json
import math

import numpy as np

from mpc_landing.mqtt.parser import RigidBodyTracker


def synth_figure8(duration: float, dt: float, radius: float, speed: float,
                  noise_std: float, seed: int = 0):
    """Generate a noisy figure-8 trajectory in the XZ plane (OptiTrack frame).

    Returns (timestamps, true_pos, true_vel, meas_pos) as (T,), (T,3), (T,3), (T,3).
    """
    rng = np.random.default_rng(seed)
    n = int(duration / dt)
    ts = np.arange(n) * dt

    # Parametric figure-8 (lemniscate-like): x = R sin(ωt), z = R sin(ωt)cos(ωt).
    # omega chosen so path speed ≈ `speed` at the centre of the lobes.
    omega = speed / radius
    x = radius * np.sin(omega * ts)
    z = radius * np.sin(omega * ts) * np.cos(omega * ts)
    y = np.zeros_like(ts)
    true_pos = np.stack([x, y, z], axis=1)

    vx = radius * omega * np.cos(omega * ts)
    vz = radius * omega * (np.cos(omega * ts) ** 2 - np.sin(omega * ts) ** 2)
    vy = np.zeros_like(ts)
    true_vel = np.stack([vx, vy, vz], axis=1)

    meas_pos = true_pos + rng.normal(0.0, noise_std, size=true_pos.shape)
    return ts, true_pos, true_vel, meas_pos


def make_payload(pos, timestamp):
    return json.dumps({
        "pos": list(pos),
        "rot": [0.0, 0.0, 0.0, 1.0],
        "metadata": {"motive_timestamp": timestamp},
    })


def finite_diff_velocity(ts, meas_pos):
    """Equivalent of the old tracker: one-step backward difference."""
    vel = np.zeros_like(meas_pos)
    for k in range(1, len(ts)):
        dt = ts[k] - ts[k - 1]
        if dt > 0:
            vel[k] = (meas_pos[k] - meas_pos[k - 1]) / dt
    return vel


def run_kalman(ts, meas_pos):
    tracker = RigidBodyTracker()
    est_vel = np.zeros_like(meas_pos)
    for k, t in enumerate(ts):
        rb = tracker.update(make_payload(meas_pos[k], t))
        est_vel[k] = rb.vel
    return est_vel


def rms(a):
    return float(np.sqrt(np.mean(a ** 2)))


def peak_xcorr_lag(true, est, dt, max_lag=20):
    """Lag (in samples) at which est's cross-correlation with true peaks.
    Positive = est lags true; 0 = aligned."""
    true = true - true.mean()
    est = est - est.mean()
    # Skip the first few samples while the filter converges.
    true = true[10:]
    est = est[10:]
    lags = range(-max_lag, max_lag + 1)
    corrs = []
    for L in lags:
        if L >= 0:
            a, b = true[L:], est[: len(true) - L]
        else:
            a, b = true[: len(true) + L], est[-L:]
        denom = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2)) + 1e-12
        corrs.append(float(np.sum(a * b) / denom))
    best = int(np.argmax(corrs))
    return lags[best], corrs[best]


def main():
    dt = 1.0 / 120.0  # 120 Hz OptiTrack
    ts, true_pos, true_vel, meas_pos = synth_figure8(
        duration=10.0, dt=dt, radius=0.9, speed=0.5, noise_std=0.003, seed=1,
    )

    fd_vel = finite_diff_velocity(ts, meas_pos)
    kf_vel = run_kalman(ts, meas_pos)

    # Skip transient at the start for a fair comparison.
    mask = slice(30, None)
    print("=== velocity RMS error (m/s) ===")
    for axis, name in enumerate("xyz"):
        e_fd = rms(fd_vel[mask, axis] - true_vel[mask, axis])
        e_kf = rms(kf_vel[mask, axis] - true_vel[mask, axis])
        ratio = e_fd / max(e_kf, 1e-9)
        print(f"  {name}:  finite-diff={e_fd:.4f}   kalman={e_kf:.4f}   fd/kf={ratio:5.1f}×")

    print("\n=== velocity lag (samples @ 120 Hz, positive = laggy) ===")
    for axis, name in [(0, "x"), (2, "z")]:  # y vel is zero, nothing to correlate
        lag_fd, _ = peak_xcorr_lag(true_vel[:, axis], fd_vel[:, axis], dt)
        lag_kf, _ = peak_xcorr_lag(true_vel[:, axis], kf_vel[:, axis], dt)
        print(f"  {name}:  finite-diff={lag_fd:+d}   kalman={lag_kf:+d}")

    # Fail-loud assertions so `python test_parser.py` doubles as a regression test.
    for axis in range(3):
        if axis == 1:  # y is zero velocity, no test
            continue
        e_fd = rms(fd_vel[mask, axis] - true_vel[mask, axis])
        e_kf = rms(kf_vel[mask, axis] - true_vel[mask, axis])
        assert e_kf < e_fd, f"Kalman should beat finite-diff on axis {axis}: {e_kf} vs {e_fd}"

    print("\nOK — Kalman beats finite-diff on both non-trivial axes.")


if __name__ == "__main__":
    main()
