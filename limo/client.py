"""UDP client for the LIMO motion-command server (`limo/server.py`).

The server runs on the robot, binds 0.0.0.0:12346, and parses newline-terminated
`linear_vel,steering` ASCII datagrams. See `limo/server.py` for the server side.

`steering` semantics depend on the LIMO chassis mode: rad/s in differential
mode, radians in Ackermann mode. Same wire value, robot decides.
"""

import socket

MAX_LINEAR_VEL = 0.5   # m/s — matches server default cap
MAX_STEERING = 1.5     # rad/s or rad, depending on chassis mode


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class LimoClient:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.sock: socket.socket | None = None

    def __enter__(self) -> "LimoClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self.send_command(0.0, 0.0)
        except Exception:
            pass
        self.close()

    def connect(self) -> None:
        """UDP is connectionless; this just creates the socket.

        `socket.connect()` on UDP sets the default destination so `send()` works
        without re-specifying the address every call.
        """
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.connect((self.host, self.port))

    def send_command(self, linear_vel: float, steering: float) -> None:
        if self.sock is None:
            raise RuntimeError("LimoClient socket is not connected")
        linear_vel = _clamp(linear_vel, -MAX_LINEAR_VEL, MAX_LINEAR_VEL)
        steering = _clamp(steering, -MAX_STEERING, MAX_STEERING)
        self.sock.send(f"{linear_vel:.4f},{steering:.4f}\n".encode())

    def close(self) -> None:
        if self.sock is not None:
            self.sock.close()
            self.sock = None
