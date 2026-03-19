"""
Linear MPC controller for quadrotor tracking and landing.

Model: 3D double integrator
    State:  [x, vx, y, vy, z, vz]  (6 states)
    Input:  [ax, ay, az]            (3 inputs = desired accelerations)

The MPC solves a finite-horizon QP at each timestep, then only the first
control input is applied. Accelerations are converted to pitch/roll/throttle
outside this module.

TODO: implement using cvxpy
"""
