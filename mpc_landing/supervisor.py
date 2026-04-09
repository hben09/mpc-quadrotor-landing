"""Boundary supervisor that wraps cf.commander to enforce arena safety.

Drop-in replacement for cf.commander — checks drone position via MQTT
on every send_setpoint() call and disarms if a boundary is violated.
"""

import time
from threading import Lock

import paho.mqtt.client as mqtt

from mpc_landing.boundary import check_boundary
from mpc_landing.mqtt.parser import RigidBodyTracker

BROKER = "rasticvm.internal"
PORT = 1883
TOPIC = "rb/crazyflie"
STALE_TIMEOUT = 2.0  # seconds without position update before forced disarm
AIRBORNE_ALT = 0.3   # metres (Y axis) — boundary checking starts after takeoff


class SafeCommander:
    """Wraps a cflib Commander to enforce arena boundary safety.

    Monitors drone position via MQTT in a background thread. Every call to
    send_setpoint() checks the latest position against arena boundaries.
    If the drone is near a wall or position data goes stale, motors are
    stopped and all further setpoint commands are blocked.

    Usage::

        with SafeCommander(cf.commander) as commander:
            commander.send_setpoint(roll, pitch, yawrate, thrust)
    """

    def __init__(self, commander, broker=BROKER, port=PORT, topic=TOPIC,
                 stale_timeout=STALE_TIMEOUT):
        self._commander = commander
        self._broker = broker
        self._port = port
        self._topic = topic
        self._stale_timeout = stale_timeout

        self._tracker = RigidBodyTracker()
        self._lock = Lock()
        self._violated = False
        self._position = None
        self._last_update = 0.0
        self._connected = False
        self._mqtt_failed = False
        self._airborne = False

        self._client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id="boundary-supervisor",
            protocol=mqtt.MQTTv311,
        )
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message

    # -- MQTT callbacks (run in background thread) --

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        with self._lock:
            self._connected = True
        client.subscribe(self._topic)

    def _on_message(self, client, userdata, msg):
        rb = self._tracker.update(msg.payload.decode())
        with self._lock:
            self._position = rb.pos
            self._last_update = time.monotonic()
            if not self._airborne and rb.pos[1] > AIRBORNE_ALT:
                self._airborne = True
            if self._airborne and not self._violated and check_boundary(rb.pos):
                self._violated = True
                self._commander.send_stop_setpoint()
                print(f"\n*** BOUNDARY VIOLATED at pos={rb.pos} — MOTORS STOPPED ***")

    # -- Commander proxy methods --

    def send_setpoint(self, roll, pitch, yawrate, thrust):
        """Send setpoint if boundary is safe, otherwise send stop."""
        with self._lock:
            if self._violated:
                self._commander.send_stop_setpoint()
                return
            # Check stale data: only relevant once airborne
            if (self._airborne
                    and time.monotonic() - self._last_update > self._stale_timeout):
                self._violated = True
                self._commander.send_stop_setpoint()
                print("\n*** POSITION DATA STALE — MOTORS STOPPED ***")
                return
        self._commander.send_setpoint(roll, pitch, yawrate, thrust)

    def send_stop_setpoint(self):
        self._commander.send_stop_setpoint()

    def send_notify_setpoint_stop(self, remain_valid_milliseconds=0):
        self._commander.send_notify_setpoint_stop(remain_valid_milliseconds)

    # -- Public properties --

    @property
    def boundary_violated(self):
        with self._lock:
            return self._violated

    @property
    def position(self):
        with self._lock:
            return self._position

    @property
    def connected(self):
        with self._lock:
            return self._connected

    # -- Lifecycle --

    def start(self):
        try:
            self._client.connect(self._broker, self._port)
            self._client.loop_start()
        except (ConnectionRefusedError, OSError) as e:
            self._mqtt_failed = True
            print(f"WARNING: MQTT unavailable ({e}) — boundary supervisor disabled")

    def stop(self):
        if not self._mqtt_failed:
            self._client.loop_stop()
            self._client.disconnect()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
