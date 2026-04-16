"""Keyboard teleoperation for a RASTIC LIMO.

W/S = forward/back | A/D = turn left/right | Space = stop | Esc = quit
"""

import argparse
import math
import sys
import time
from threading import Lock

from pynput import keyboard

from limo.client import MAX_LINEAR_VEL, MAX_STEERING, LimoClient
from limo.pose import DEFAULT_BROKER, DEFAULT_MQTT_PORT, LimoPoseSubscriber, topic_for
from limo.registry import DEFAULT_LIMO, TCP_PORT, ip_for

CONTROL_HZ = 50.0
CONTROL_DT = 1.0 / CONTROL_HZ

_keys: set = set()
_keys_lock = Lock()
_running = True


def _on_press(key):
    global _running
    if key == keyboard.Key.esc:
        _running = False
        return False
    with _keys_lock:
        try:
            _keys.add(key.char.lower())
        except AttributeError:
            _keys.add(key)


def _on_release(key):
    with _keys_lock:
        try:
            _keys.discard(key.char.lower())
        except AttributeError:
            _keys.discard(key)


def _compute_command() -> tuple[float, float]:
    with _keys_lock:
        keys = set(_keys)

    if keyboard.Key.space in keys:
        return 0.0, 0.0

    linear = 0.0
    steering = 0.0
    if "w" in keys:
        linear += MAX_LINEAR_VEL
    if "s" in keys:
        linear -= MAX_LINEAR_VEL
    if "a" in keys:
        steering += MAX_STEERING
    if "d" in keys:
        steering -= MAX_STEERING
    return linear, steering


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyboard teleop for a RASTIC LIMO")
    parser.add_argument("--limo", default=DEFAULT_LIMO,
                        help="LIMO ID (number on the robot). Resolves IP via registry.")
    parser.add_argument("--ip", default=None,
                        help="Override TCP IP (skips registry lookup)")
    parser.add_argument("--port", type=int, default=TCP_PORT, help="TCP command port")
    parser.add_argument("--broker", default=DEFAULT_BROKER, help="MQTT broker hostname")
    parser.add_argument("--mqtt-port", type=int, default=DEFAULT_MQTT_PORT)
    parser.add_argument("--mqtt-topic", default=None,
                        help="MQTT pose topic. Default rb/limo<id>; "
                             "set to rb/landing when the landing platform is mounted.")
    args = parser.parse_args()

    ip = args.ip if args.ip is not None else ip_for(args.limo)
    topic = args.mqtt_topic if args.mqtt_topic is not None else topic_for(args.limo)

    pose = LimoPoseSubscriber(args.broker, args.mqtt_port, topic)
    pose.connect()

    print(f"LIMO {args.limo} @ {ip}:{args.port}  |  pose topic: {topic}")
    print("W/S = forward/back | A/D = turn | Space = stop | Esc = quit")

    listener = keyboard.Listener(on_press=_on_press, on_release=_on_release, suppress=True)
    listener.start()

    try:
        with LimoClient(ip, args.port) as robot:
            while _running:
                linear, steering = _compute_command()
                robot.send_command(linear, steering)

                snap = pose.latest()
                if snap is None:
                    pose_str = "[no pose]"
                else:
                    rb, age = snap
                    x, y, z = rb.pos
                    pose_str = (f"x={x:+.2f} y={y:+.2f} z={z:+.2f} "
                                f"yaw={math.degrees(rb.yaw):+6.1f}deg age={age:4.2f}s")

                sys.stdout.write(
                    f"\rcmd v={linear:+.2f} s={steering:+.2f}  |  {pose_str}    "
                )
                sys.stdout.flush()
                time.sleep(CONTROL_DT)
    finally:
        listener.stop()
        pose.stop()
        print("\nStopped.")


if __name__ == "__main__":
    main()
