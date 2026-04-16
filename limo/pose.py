"""MQTT pose subscriber for a LIMO rigid body.

Reuses `mpc_landing.mqtt.parser.RigidBodyTracker` so velocity finite-differencing
matches the Crazyflie pipeline. Topic defaults to `rb/limo<id>` for the bare
robot; pass `--mqtt-topic rb/landing` once the landing platform (with its own
markers) is mounted.
"""

import threading
import time

import paho.mqtt.client as mqtt

from mpc_landing.mqtt.parser import MQTTRigidBody, RigidBodyTracker

DEFAULT_BROKER = "rasticvm.internal"
DEFAULT_MQTT_PORT = 1883


def topic_for(limo_id: str) -> str:
    return f"rb/limo{limo_id}"


class LimoPoseSubscriber:
    def __init__(self, broker: str, port: int, topic: str):
        self.broker = broker
        self.port = port
        self.topic = topic

        self._tracker = RigidBodyTracker()
        self._lock = threading.Lock()
        self._latest: MQTTRigidBody | None = None
        self._last_update: float = 0.0

        self.client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=f"limo-pose-{topic.replace('/', '-')}",
            protocol=mqtt.MQTTv311,
        )
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        client.subscribe(self.topic)

    def _on_message(self, client, userdata, msg):
        try:
            rb = self._tracker.update(msg.payload.decode())
        except Exception:
            return
        with self._lock:
            self._latest = rb
            self._last_update = time.time()

    def connect(self) -> None:
        """Best-effort connect. Returns immediately; reconnects in background.

        If the broker is unreachable (e.g. running off-site without RASTIC
        access), `latest()` simply keeps returning None and the rest of the
        pipeline still works.
        """
        self.client.connect_async(self.broker, self.port, 60)
        self.client.loop_start()

    def stop(self) -> None:
        self.client.loop_stop()
        self.client.disconnect()

    def latest(self) -> tuple[MQTTRigidBody, float] | None:
        """Return (rigid body, age in seconds) or None if no message received yet."""
        with self._lock:
            if self._latest is None:
                return None
            return self._latest, time.time() - self._last_update
