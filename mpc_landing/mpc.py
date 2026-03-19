"""
Linear MPC controller for quadrotor tracking and landing.

Model: 3D double integrator (decoupled per axis)
    State:  [px, vx, py, vy, pz, vz]  (6 states)
    Input:  [ax, ay, az]               (3 inputs = desired accelerations)

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

    dt: float = 0.02            # timestep (s), 50Hz
    horizon: int = 25           # prediction horizon (N steps = 0.5s at 50Hz)

    # State cost weights: [px, vx, py, vy, pz, vz]
    Q_diag: list = field(default_factory=lambda: [10.0, 1.0, 10.0, 1.0, 10.0, 1.0])

    # Terminal cost weights (larger than Q to incentivize reaching target)
    Qf_diag: list = field(default_factory=lambda: [100.0, 10.0, 100.0, 10.0, 100.0, 10.0])

    # Input cost weights: [ax, ay, az]
    R_diag: list = field(default_factory=lambda: [1.0, 1.0, 1.0])

    # Acceleration limits (m/s^2)
    a_max: float = 5.0

    # Velocity limits (m/s) — set None to disable
    v_max: float = 2.0


class MPCController:
    def __init__(self, config: MPCConfig = None):
        self.cfg = config or MPCConfig()
        self._build_dynamics()
        self._build_problem()

    def _build_dynamics(self):
        """Construct discrete-time A, B matrices for 3D double integrator."""
        dt = self.cfg.dt
        nx, nu = 6, 3

        self.A = np.eye(nx)
        self.B = np.zeros((nx, nu))
        for i in range(3):
            self.A[2*i, 2*i+1] = dt
            self.B[2*i, i] = 0.5 * dt**2
            self.B[2*i+1, i] = dt

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

        Q = np.diag(self.cfg.Q_diag)
        Qf = np.diag(self.cfg.Qf_diag)
        R = np.diag(self.cfg.R_diag)

        cost = 0
        constraints = [self.x_var[0] == self.x0_param]

        for k in range(N):
            err = self.x_var[k] - self.ref_param[k]
            cost += cp.quad_form(err, Q)
            cost += cp.quad_form(self.u_var[k], R)

            # Dynamics
            constraints.append(
                self.x_var[k + 1] == self.A @ self.x_var[k] + self.B @ self.u_var[k]
            )

            # Acceleration limits
            constraints.append(self.u_var[k] <= self.cfg.a_max)
            constraints.append(self.u_var[k] >= -self.cfg.a_max)

            # Velocity limits
            if self.cfg.v_max is not None:
                for i in range(3):
                    constraints.append(self.x_var[k + 1, 2*i+1] <= self.cfg.v_max)
                    constraints.append(self.x_var[k + 1, 2*i+1] >= -self.cfg.v_max)

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
        self.x0_param.value = x0
        self.ref_param.value = reference

        try:
            self.problem.solve(solver=cp.OSQP, warm_start=True)

            if self.problem.status in ('optimal', 'optimal_inaccurate'):
                return self.u_var.value[0]
            else:
                print(f"MPC solver status: {self.problem.status}")
                return np.zeros(3)
        except cp.SolverError as e:
            print(f"MPC solver error: {e}")
            return np.zeros(3)

    def get_planned_trajectory(self):
        """Return the full planned state trajectory (useful for visualization)."""
        if self.x_var.value is not None:
            return self.x_var.value
        return None

    def get_planned_inputs(self):
        """Return the full planned input sequence (useful for visualization)."""
        if self.u_var.value is not None:
            return self.u_var.value
        return None
