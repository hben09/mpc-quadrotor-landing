"""Keyboard teleoperation for a RASTIC LIMO.

W/S = forward/back | A/D = turn left/right | Space = stop
C   = toggle circle mode   (fixed-radius loop)
F   = toggle figure-8 mode (sinusoidal steering, same lobe size as circle)
Esc = quit

Any WASD or Space input cancels autonomous modes.
"""

import argparse
import math
import sys
import time
from threading import Lock

from pynput import keyboard

from limo.client import MAX_LINEAR_VEL, MAX_STEERING, LimoClient
from limo.pose import DEFAULT_BROKER, DEFAULT_MQTT_PORT, LimoPoseSubscriber, topic_for
from limo.registry import DEFAULT_LIMO, UDP_PORT, ip_for

CONTROL_HZ = 50.0
CONTROL_DT = 1.0 / CONTROL_HZ

DEFAULT_CIRCLE_DIAMETER = 1.8   # metres
DEFAULT_WHEELBASE = 0.20        # metres — LIMO Pro spec; tune if circles are off

_keys: set = set()
_keys_lock = Lock()
_running = True
_circle_on = False
_figure8_on = False
_figure8_t0 = 0.0
_mode_lock = Lock()


def _on_press(key):
    global _running, _circle_on, _figure8_on, _figure8_t0
    if key == keyboard.Key.esc:
        _running = False
        return False
    char = getattr(key, "char", None)
    char = char.lower() if char else None
    if char == "c":
        with _mode_lock:
            _circle_on = not _circle_on
            if _circle_on:
                _figure8_on = False
        return
    if char == "f":
        with _mode_lock:
            _figure8_on = not _figure8_on
            if _figure8_on:
                _circle_on = False
                _figure8_t0 = time.time()
        return
    with _keys_lock:
        if char is not None:
            _keys.add(char)
        else:
            _keys.add(key)


def _on_release(key):
    with _keys_lock:
        try:
            _keys.discard(key.char.lower())
        except AttributeError:
            _keys.discard(key)


def _compute_command(circle_diameter: float, wheelbase: float) -> tuple[float, float, str]:
    """Return (linear_vel, steering, mode_label)."""
    global _circle_on, _figure8_on
    with _keys_lock:
        keys = set(_keys)

    manual_pressed = bool(keys & {"w", "s", "a", "d", keyboard.Key.space})

    # Any manual input cancels autonomous modes.
    if manual_pressed:
        with _mode_lock:
            _circle_on = False
            _figure8_on = False

    with _mode_lock:
        circle_active = _circle_on
        figure8_active = _figure8_on
        f8_t0 = _figure8_t0

    radius = circle_diameter / 2.0
    # Ackermann bicycle model: δ = atan(L / R). Positive = CCW.
    max_steer = math.atan(wheelbase / radius)

    if circle_active:
        return MAX_LINEAR_VEL, max_steer, "CIRCLE"

    if figure8_active:
        # Two tangent circles: CCW loop, then CW loop. Each loop takes one full
        # circumference at max speed → T_loop = 2πR/v. Full figure-8 = 2×T_loop.
        t_loop = 2.0 * math.pi * radius / MAX_LINEAR_VEL
        t = time.time() - f8_t0
        phase = (t % (2.0 * t_loop)) / t_loop   # 0.0–2.0
        steering = max_steer if phase < 1.0 else -max_steer
        return MAX_LINEAR_VEL, steering, "FIG-8"

    if keyboard.Key.space in keys:
        return 0.0, 0.0, "MANUAL"

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
    return linear, steering, "MANUAL"


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyboard teleop for a RASTIC LIMO")
    parser.add_argument("--limo", default=DEFAULT_LIMO,
                        help="LIMO ID (number on the robot). Resolves IP via registry.")
    parser.add_argument("--ip", default=None,
                        help="Override TCP IP (skips registry lookup)")
    parser.add_argument("--port", type=int, default=UDP_PORT, help="UDP command port")
    parser.add_argument("--broker", default=DEFAULT_BROKER, help="MQTT broker hostname")
    parser.add_argument("--mqtt-port", type=int, default=DEFAULT_MQTT_PORT)
    parser.add_argument("--mqtt-topic", default=None,
                        help="MQTT pose topic. Default rb/limo<id>; "
                             "set to rb/landing when the landing platform is mounted.")
    parser.add_argument("--circle-diameter", type=float, default=DEFAULT_CIRCLE_DIAMETER,
                        help="Circle diameter (m) when circle mode is on (press C).")
    parser.add_argument("--wheelbase", type=float, default=DEFAULT_WHEELBASE,
                        help="LIMO wheelbase (m). Used for Ackermann circle geometry.")
    args = parser.parse_args()

    ip = args.ip if args.ip is not None else ip_for(args.limo)
    topic = args.mqtt_topic if args.mqtt_topic is not None else topic_for(args.limo)

    pose = LimoPoseSubscriber(args.broker, args.mqtt_port, topic)
    pose.connect()

    print(f"LIMO {args.limo} @ {ip}:{args.port}  |  pose topic: {topic}")
    print(f"W/S = fwd/back | A/D = turn | Space = stop")
    t_loop = 2.0 * math.pi * (args.circle_diameter / 2.0) / MAX_LINEAR_VEL
    print(f"C = circle ({args.circle_diameter:.1f} m dia) | "
          f"F = figure-8 ({2 * t_loop:.1f}s/cycle) | Esc = quit")

    listener = keyboard.Listener(on_press=_on_press, on_release=_on_release, suppress=True)
    listener.start()

    try:
        with LimoClient(ip, args.port) as robot:
            while _running:
                linear, steering, mode_str = _compute_command(
                    args.circle_diameter, args.wheelbase)
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
                    f"\r[{mode_str:7s}] v={linear:+.2f} s={steering:+.2f}  |  {pose_str}    "
                )
                sys.stdout.flush()
                time.sleep(CONTROL_DT)
    finally:
        listener.stop()
        pose.stop()
        print("\nStopped.")


if __name__ == "__main__":
    main()
