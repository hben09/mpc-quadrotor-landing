"""
Offset-free MPC controller for quadrotor tracking and landing.

Model: 3D double integrator with gravity + vertical disturbance
    State:  [px, vx, py, vy, pz, vz, d]  (7 states, d = vertical disturbance)
    Input:  [ax, ay, az]                  (3 inputs = desired accelerations)

The disturbance state `d` is a constant bias on vertical acceleration,
estimated each step from the prediction error. It absorbs model mismatch
(battery drift, weight changes) so the optimizer plans around it —
eliminating steady-state altitude error without an external integrator.

Solves a finite-horizon QP at each timestep. Only the first control input
is applied. Accelerations are converted to pitch/roll/throttle outside
this module.

No ROS, no sim dependencies — just numpy and cvxpy.
"""

import numpy as np
import cvxpy as cp
from dataclasses import dataclass, field


@dataclass
class MPCConfig:
    """All tunable MPC parameters in one place."""

    dt: float = 0.02  # timestep (s), 50Hz
    horizon: int = 25  # prediction horizon (N steps = 0.5s at 50Hz)

    # State cost weights: [px, vx, py, vy, pz, vz]
    Q_diag: list = field(default_factory=lambda: [10.0, 1.0, 10.0, 1.0, 10.0, 1.0])

    # Terminal cost weights (larger than Q to incentivize reaching target)
    Qf_diag: list = field(
        default_factory=lambda: [200.0, 20.0, 100.0, 10.0, 100.0, 10.0]
    )

    # Input cost weights: [ax, ay, az]
    R_diag: list = field(default_factory=lambda: [1.0, 1.0, 1.0])

    # Acceleration limits (m/s^2)
    a_max: float = 15.0

    # Velocity limits (m/s) — set None to disable
    v_max: float = 2.0


class MPCController:
    def __init__(self, config: MPCConfig = None):
        self.cfg = config or MPCConfig()
        self._build_dynamics()
        self._build_problem()

    def _build_dynamics(self):
        """Construct discrete-time A, B matrices for 3D double integrator + disturbance."""
        dt = self.cfg.dt
        nx, nu = 7, 3  # 6 physical states + 1 disturbance

        # Physical double integrator (first 6 states)
        self.A = np.eye(nx)
        self.B = np.zeros((nx, nu))
        for i in range(3):
            self.A[2 * i, 2 * i + 1] = dt
            self.B[2 * i, i] = 0.5 * dt**2
            self.B[2 * i + 1, i] = dt

        # Disturbance state d (idx 6): constant bias on vertical acceleration
        # d[k+1] = d[k]  (already 1.0 on diagonal from np.eye)
        # d affects py and vy the same way as ay
        self.A[2, 6] = 0.5 * dt**2  # py += 0.5*d*dt^2
        self.A[3, 6] = dt  # vy += d*dt

        # Gravity affine term: only affects py (idx 2) and vy (idx 3)
        self.g_vec = np.zeros(nx)
        self.g_vec[2] = -0.5 * 9.81 * dt**2
        self.g_vec[3] = -9.81 * dt

        # Equilibrium input: ay = G to counteract gravity (hover = zero R cost)
        self.u_eq = np.zeros(nu)
        self.u_eq[1] = 9.81

        # Disturbance estimator gain (simple single-step correction)
        self._d_hat = 0.0
        self._x_pred = None
        self._last_status = ""

        self.nx = nx
        self.nu = nu

    def _build_problem(self):
        """Build the cvxpy problem with parameters (solved repeatedly at runtime)."""
        N = self.cfg.horizon
        nx, nu = self.nx, self.nu

        # Decision variables
        self.x_var = cp.Variable((N + 1, nx))
        self.u_var = cp.Variable((N, nu))

        # Parameters (updated each call to compute())
        self.x0_param = cp.Parameter(nx)
        self.ref_param = cp.Parameter((N + 1, nx))

        # Extend Q/Qf with zero weight on disturbance state (idx 6)
        Q = np.diag(self.cfg.Q_diag + [0.0])
        Qf = np.diag(self.cfg.Qf_diag + [0.0])
        R = np.diag(self.cfg.R_diag)

        cost = 0
        constraints = [self.x_var[0] == self.x0_param]

        for k in range(N):
            err = self.x_var[k] - self.ref_param[k]
            cost += cp.quad_form(err, Q)
            cost += cp.quad_form(self.u_var[k] - self.u_eq, R)

            # Dynamics
            constraints.append(
                self.x_var[k + 1]
                == self.A @ self.x_var[k] + self.B @ self.u_var[k] + self.g_vec
            )

            # Acceleration limits
            constraints.append(self.u_var[k] <= self.cfg.a_max)
            constraints.append(self.u_var[k] >= -self.cfg.a_max)

            # Velocity limits
            if self.cfg.v_max is not None:
                for i in range(3):
                    constraints.append(self.x_var[k + 1, 2 * i + 1] <= self.cfg.v_max)
                    constraints.append(self.x_var[k + 1, 2 * i + 1] >= -self.cfg.v_max)

        # Terminal cost
        err_f = self.x_var[N] - self.ref_param[N]
        cost += cp.quad_form(err_f, Qf)

        self.problem = cp.Problem(cp.Minimize(cost), constraints)

    def compute(self, x0, reference):
        """Solve the MPC QP.

        Args:
            x0: np.array of shape (6,) — current state [px, vx, py, vy, pz, vz]
            reference: np.array of shape (N+1, 6) — reference trajectory over horizon

        Returns:
            np.array([ax, ay, az]) — first optimal control input.
            Returns zeros if the solver fails.
        """
        # Update disturbance estimate from prediction error
        if self._x_pred is not None:
            # Prediction error on vy (index 3) — most responsive to thrust bias
            vy_error = x0[3] - self._x_pred[3]
            # Low-pass update: d_hat += alpha * (error / dt)
            self._d_hat += 0.3 * (vy_error / self.cfg.dt)

        # Augment state with disturbance estimate
        x0_aug = np.append(x0, self._d_hat)

        # Augment reference with zero disturbance target
        ref_aug = np.zeros((reference.shape[0], self.nx))
        ref_aug[:, :6] = reference

        self.x0_param.value = x0_aug
        self.ref_param.value = ref_aug

        try:
            self.problem.solve(solver=cp.OSQP, warm_start=True)
            self._last_status = self.problem.status

            if self.problem.status in ("optimal", "optimal_inaccurate"):
                # Store predicted next state for disturbance estimation
                self._x_pred = self.x_var.value[1, :6]
                return self.u_var.value[0]
            else:
                print(f"MPC solver status: {self.problem.status}")
                self._x_pred = None
                return np.zeros(3)
        except cp.SolverError as e:
            print(f"MPC solver error: {e}")
            self._last_status = "solver_error"
            self._x_pred = None
            return np.zeros(3)

    def get_planned_trajectory(self):
        """Return the full planned state trajectory (useful for visualization)."""
        if self.x_var.value is not None:
            return self.x_var.value[:, :6]  # exclude disturbance state
        return None

    @property
    def disturbance(self):
        """Current disturbance estimate (m/s^2 bias on vertical acceleration)."""
        return self._d_hat

    @property
    def last_status(self):
        return self._last_status

    def get_planned_inputs(self):
        """Return the full planned input sequence (useful for visualization)."""
        if self.u_var.value is not None:
            return self.u_var.value
        return None
