# LIMO ground-vehicle control

Keyboard + autonomous-pattern control for RASTIC's AgileX LIMO robots. Part of the `mpc-quadrotor-landing` project — the LIMO is (eventually) the moving platform the quadrotor lands on.

## Why our own stack (not the class's `limoRemoteControl.py`)

The class script has three properties that hurt a real-time control loop:

1. **TCP with per-command ack.** The server sends an ASCII response per command. A client that doesn't drain the response buffer causes kernel-level TCP backpressure, which stalls the server's `send()` and blocks it from receiving new commands. Symptom: commands appear to lag by several seconds under sustained use.
2. **300 ms staleness timeout.** Our wifi ping data shows max RTT = 300 ms and stddev = 72 ms on a nominally-0%-loss link. Any RTT spike above the cutoff zeros the linear velocity, causing intermittent robot stops.
3. **Hardcoded 0.2 m/s cap.**

This package ships a drop-in replacement: a UDP server on port 12346 (co-exists with the class's TCP 12345), 1 s staleness window, no acks, configurable speed cap.

## Layout

```
limo/
  server.py    # runs ON the robot — UDP motion-command server
  client.py    # runs on your laptop — UDP client wrapper
  pose.py      # MQTT subscriber for rb/limoXXX pose from OptiTrack
  registry.py  # LIMO ID → IP mapping
  teleop.py    # keyboard teleop + autonomous patterns (circle, figure-8)
```

## First-time setup

### 1. Network

You need routing to the LIMO's `172.16.x.x` IP. At RASTIC this means the **ME416** wifi (password `agileagile`). Confirm:

```bash
ping -c 2 <LIMO_IP>
```

### 2. Deploy the server

```bash
scp limo/server.py agilex@<LIMO_IP>:~/limo_server.py
# password: agx
```

The server uses `pylimo` which is preinstalled on the robot — no dependencies to install.

### 3. Start the server on the robot

```bash
ssh agilex@<LIMO_IP>
python3 limo_server.py
# → LIMO UDP server listening on 0.0.0.0:12346 (staleness=1.0s, max_linear=0.5 m/s)
```

Leave this session open. Override defaults if needed:

```bash
python3 limo_server.py --max-linear 0.8 --staleness 2.0
```

### 4. Register the LIMO on your laptop

Edit [limo/registry.py](registry.py) and add your robot's number → IP. Find the IP via `ping limoXXX` on the ME416 wifi.

## Running teleop

```bash
uv run limo-teleop                    # defaults to DEFAULT_LIMO
uv run limo-teleop --limo 814         # different robot
```

### Controls

| Key     | Action                                         |
|---------|------------------------------------------------|
| W / S   | Forward / back                                 |
| A / D   | Turn left / right                              |
| Space   | Stop                                           |
| C       | Toggle circle mode (constant-curvature loop)   |
| F       | Toggle figure-8 mode (sinusoidal steering)     |
| Esc     | Quit                                           |

Any WASD or Space press cancels the autonomous mode.

### Autonomous patterns

**Circle** (`C`): Drives a constant-radius loop using the Ackermann bicycle model `δ = atan(L / R)`. Defaults to 1.8 m diameter; tune with `--circle-diameter` and `--wheelbase` if the observed circle differs from requested.

**Figure-8** (`F`): Two tangent circles at the start point — CCW loop, then CW loop, repeat. Steering alternates between `+δ_max` and `-δ_max`. Crossover happens wherever you press F. Period auto-computed from `--circle-diameter`: `T = 4πR/v`. Each lobe matches the `C`-mode circle.

## Pose feedback

`pose.py` subscribes to `rb/limo<id>` on the RASTIC MQTT broker (`rasticvm.internal:1883`) and reuses `mpc_landing.mqtt.parser.RigidBodyTracker` for velocity finite-differencing — so the LIMO pose pipeline is consistent with the Crazyflie's.

When the landing platform is physically mounted on the LIMO, switch the topic:

```bash
uv run limo-teleop --mqtt-topic rb/landing
```

This makes the LIMO a drop-in moving target for `hardware/mpc_teleop_landing.py` without changes to the drone-side code.

## Wire format (for debugging without the client)

Server accepts UDP datagrams of the form:

```
linear_vel,steering\n
```

ASCII, comma-separated. Debug with netcat:

```bash
nc -u <LIMO_IP> 12346
# type: 0.2,0.0   and press enter
```

## Chassis modes

LIMOs have differential and Ackermann chassis modes. In **differential**, `steering` is interpreted as angular velocity (rad/s); in **Ackermann**, as wheel angle (rad). The `C` and `F` trajectory formulas assume **Ackermann** because that's what we observed on the ME416 robots. If the robot you're using is in differential mode, the circles will be wrongly-sized — tell me and we'll add a mode flag.

## Registry

| ID   | IP             | Notes                 |
|------|----------------|-----------------------|
| 793  | 172.16.2.92    | From class sample     |
| 809  | 172.16.1.43    | Tested, working       |
| 814  | 172.16.1.136   | Not ME416-tagged      |

Add new robots in [registry.py](registry.py).
