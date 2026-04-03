# Observation

1. The difference between the drone and the ground bot on the x-axis approaches `-0.2` as time increases. Under ideal conditions, this difference should be `0`.

# Tuning Actions

1. Increase `K_PITCH` (`PWM/(m/s^2)`), since the pitch angle controls acceleration on the x-axis.
2. Increase `MAX_TILT_PWM` to allow more aggressive dynamics.
3. Increase `px` and `vx` in the terminal weight (`Qf_diag`).
    I doubled them from `px = 100, vx = 10` to `px = 200, vx = 20`.
    This resulted in a successful landing, although a delta in x still exists.

    Before changing `Qf_diag`:
    `sim/plot/run_20260403_154949_409695.png`

    After changing `Qf_diag`:
    `sim/plot/run_20260403_155607_086406.png`

    Notice the difference after `DISARM`.

4. Increase planning horizon N. (before: sim/data/run_20260403_155607_086406)
(after: sim/data/run_20260403_161547_583123)
    N increased from 20 to 40, while other parameters remains same.
    Final difference in x axis is furthur dereased.
    See figure (sim/plot/run_20260403_161547_583123.png)
