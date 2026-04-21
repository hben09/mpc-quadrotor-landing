#!/usr/bin/env python3
"""UDP motion-command server — runs ON the LIMO, not on your laptop.

Deploy:
    scp limo/server.py agilex@<LIMO_IP>:~/limo_server.py
    ssh agilex@<LIMO_IP>
    python3 limo_server.py

Wire format (ASCII, one datagram = one command):
    "linear_vel,steering\\n"
where `steering` is rad/s in differential mode or rad in Ackermann mode.

Why a rewrite of the class's `limoRemoteControl.py`:
- UDP instead of TCP: no Nagle, no backpressure, no recv-buffer fill, no ack drain.
- 1 s staleness timeout instead of 300 ms: wifi jitter on this network hits
  300 ms routinely (observed max RTT exactly matches the old cutoff), which
  made the robot stop every few seconds.
- No per-command ack: one less round-trip, simpler failure modes.

Runs alongside the class server — different port (12346 vs 12345).
"""

import argparse
import socket
import threading
import time

try:
    from pylimo import limo
except ImportError as exc:  # pragma: no cover — diagnostic only
    raise SystemExit(
        "pylimo not found. This script must run on the LIMO robot, "
        "not on the laptop."
    ) from exc


DEFAULT_PORT = 12346
DEFAULT_STALENESS = 1.0    # s — zero linear if no command within this window
DEFAULT_MAX_LINEAR = 0.5   # m/s
CONTROL_DT = 0.05          # 20 Hz actuator loop


class Command:
    __slots__ = ("linear_vel", "steering", "timestamp")

    def __init__(self):
        self.linear_vel = 0.0
        self.steering = 0.0
        self.timestamp = 0.0


def recv_loop(sock: socket.socket, cmd: Command, lock: threading.Lock,
              max_linear: float) -> None:
    while True:
        try:
            data, _addr = sock.recvfrom(1024)
        except OSError:
            return
        try:
            linear_str, steering_str = data.decode("utf-8").strip().split(",")
            linear = max(-max_linear, min(max_linear, float(linear_str)))
            steering = float(steering_str)
        except (UnicodeDecodeError, ValueError):
            continue
        with lock:
            cmd.linear_vel = linear
            cmd.steering = steering
            cmd.timestamp = time.time()


def motion_loop(robot, cmd: Command, lock: threading.Lock,
                staleness: float) -> None:
    while True:
        with lock:
            linear = cmd.linear_vel
            steering = cmd.steering
            age = time.time() - cmd.timestamp

        if age > staleness:
            linear = 0.0  # zero forward drive; hold steering (as class server does)

        robot.SetMotionCommand(linear_vel=linear, steering_angle=steering)
        time.sleep(CONTROL_DT)


def main() -> None:
    parser = argparse.ArgumentParser(description="LIMO UDP motion-command server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--staleness", type=float, default=DEFAULT_STALENESS,
                        help="Zero linear velocity if no command within this many seconds")
    parser.add_argument("--max-linear", type=float, default=DEFAULT_MAX_LINEAR,
                        help="Clamp linear velocity magnitude (m/s)")
    args = parser.parse_args()

    robot = limo.LIMO()
    robot.EnableCommand()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", args.port))

    cmd = Command()
    lock = threading.Lock()

    print(f"LIMO UDP server listening on 0.0.0.0:{args.port}  "
          f"(staleness={args.staleness}s, max_linear={args.max_linear} m/s)")

    threading.Thread(
        target=motion_loop, args=(robot, cmd, lock, args.staleness), daemon=True,
    ).start()

    try:
        recv_loop(sock, cmd, lock, args.max_linear)
    except KeyboardInterrupt:
        pass
    finally:
        robot.SetMotionCommand(linear_vel=0.0, steering_angle=0.0)
        sock.close()
        print("\nStopped.")


if __name__ == "__main__":
    main()
