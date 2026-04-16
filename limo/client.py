"""TCP client for LIMO velocity commands.

The robot runs `udpStreamer/limoRemoteControl.py` (despite the name, it's TCP),
listening on port 12345. Each command is a newline-terminated
`linear_vel,steering` ASCII pair. The robot caps linear at 0.2 m/s and zeros it
if no command arrives for 300 ms.

`steering` semantics depend on the LIMO chassis mode: rad/s in differential
mode, radians in Ackermann mode. Same wire value, robot decides.

The server sends an ASCII ack per command. We must drain it — otherwise the
recv buffer fills, backpressure stalls the server, and commands appear to lag
by seconds. A background thread does this.
"""

import socket
import threading

MAX_LINEAR_VEL = 0.2   # m/s — matches server cap
MAX_STEERING = 1.0     # rad/s or rad, depending on chassis mode


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class LimoClient:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.sock: socket.socket | None = None
        self._drain_thread: threading.Thread | None = None
        self._stop = threading.Event()

    def __enter__(self) -> "LimoClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self.send_command(0.0, 0.0)
        except Exception:
            pass
        self.close()

    def connect(self, timeout: float = 3.0) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(timeout)
        self.sock.connect((self.host, self.port))
        self.sock.settimeout(None)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        self._stop.clear()
        self._drain_thread = threading.Thread(target=self._drain_responses, daemon=True)
        self._drain_thread.start()

    def _drain_responses(self) -> None:
        sock = self.sock
        if sock is None:
            return
        try:
            while not self._stop.is_set():
                if not sock.recv(4096):
                    return
        except OSError:
            return

    def send_command(self, linear_vel: float, steering: float) -> None:
        if self.sock is None:
            raise RuntimeError("LimoClient socket is not connected")
        linear_vel = _clamp(linear_vel, -MAX_LINEAR_VEL, MAX_LINEAR_VEL)
        steering = _clamp(steering, -MAX_STEERING, MAX_STEERING)
        self.sock.sendall(f"{linear_vel:.4f},{steering:.4f}\n".encode())

    def close(self) -> None:
        self._stop.set()
        if self.sock is not None:
            self.sock.close()
            self.sock = None
        if self._drain_thread is not None:
            self._drain_thread.join(timeout=1.0)
            self._drain_thread = None
