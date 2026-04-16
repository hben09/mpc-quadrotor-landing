"""Crazyflie battery telemetry → MQTT.

Reads pm.vbat / pm.batteryLevel / pm.state via cflib LogConfig and publishes
each sample as JSON to MQTT topic `cf/battery`, so viewers like
hardware/dashboard.py can display it.
"""

import json

from cflib.crazyflie.log import LogConfig

TOPIC = "cf/battery"
PERIOD_MS = 500  # 2 Hz — battery changes slowly


class BatteryPublisher:
    def __init__(self, cf, mqtt_client, topic: str = TOPIC, period_ms: int = PERIOD_MS):
        self._cf = cf
        self._mqtt = mqtt_client
        self._topic = topic
        self._period_ms = period_ms
        self._lc: LogConfig | None = None

    def _on_data(self, _ts, data, _logconf):
        payload = {
            "vbat": float(data["pm.vbat"]),
            "level": int(data["pm.batteryLevel"]),
            "state": int(data["pm.state"]),
        }
        self._mqtt.publish(self._topic, json.dumps(payload), retain=True)

    def _on_error(self, _logconf, msg):
        print(f"Battery log error: {msg}")

    def start(self):
        lc = LogConfig(name="Battery", period_in_ms=self._period_ms)
        lc.add_variable("pm.vbat", "float")
        lc.add_variable("pm.batteryLevel", "uint8_t")
        lc.add_variable("pm.state", "uint8_t")
        self._cf.log.add_config(lc)
        lc.data_received_cb.add_callback(self._on_data)
        lc.error_cb.add_callback(self._on_error)
        lc.start()
        self._lc = lc

    def stop(self):
        if self._lc is not None:
            try:
                self._lc.stop()
            except Exception:
                pass
            self._lc = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_exc):
        self.stop()
