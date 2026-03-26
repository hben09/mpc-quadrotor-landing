#!/usr/bin/env python3
"""Simple MQTT subscriber using MQTT 3.1.1."""

import paho.mqtt.client as mqtt

from mqtt_parser import RigidBodyTracker

BROKER = "rasticvm.internal"
PORT = 1883
TOPIC = "rb/limo777"

tracker = RigidBodyTracker()


def on_connect(client, userdata, flags, reason_code, properties):
    print(f"Connected (rc={reason_code})")
    client.subscribe(TOPIC)
    print(f"Subscribed to '{TOPIC}'")


def on_message(client, userdata, msg):
    rb = tracker.update(msg.payload.decode())
    print(
        f"[{msg.topic}] "
        f"pos={[f'{v:.4f}' for v in rb.pos]}  "
        f"euler={[f'{v:.4f}' for v in rb.euler]}  "
        f"vel={[f'{v:.4f}' for v in rb.vel]}",
        flush=True,
    )


def main():
    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id="mpc-quadrotor-sub",
        protocol=mqtt.MQTTv311,
    )
    client.on_connect = on_connect
    client.on_message = on_message

    print(f"Connecting to {BROKER}:{PORT} ...")
    client.connect(BROKER, PORT)

    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print("\nDisconnecting...")
        client.disconnect()


if __name__ == "__main__":
    main()
