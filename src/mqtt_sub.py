#!/usr/bin/env python3
"""Simple MQTT subscriber using MQTT 3.1.1."""

import paho.mqtt.client as mqtt

BROKER = "rasticvm.internal"
PORT = 1883
TOPIC = "rb/limo777"


def on_connect(client, userdata, flags, reason_code, properties):
    print(f"Connected (rc={reason_code})")
    client.subscribe(TOPIC)
    print(f"Subscribed to '{TOPIC}'")


def on_message(client, userdata, msg):
    print(f"[{msg.topic}] {msg.payload.decode()}")


def main():
    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
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
