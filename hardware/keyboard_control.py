import logging
import sys
import time
from pathlib import Path
from threading import Lock

from pynput import keyboard

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils import uri_helper

URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E7')

MAX_ROLL = 15.0       # degrees
MAX_PITCH = 15.0      # degrees
MAX_YAWRATE = 60.0    # degrees/s
MAX_THRUST = 45000

# Discrete thrust modes
MODE_DISARM = 'DISARM'
MODE_ARM = 'ARM'
MODE_HOVER = 'HOVER'
MODE_THRUST = {MODE_DISARM: 0, MODE_ARM: 15000, MODE_HOVER: 45000}
THRUST_STEP = 5000

logging.basicConfig(level=logging.ERROR)

# Shared state
pressed_keys = set()
keys_lock = Lock()
running = True


def on_press(key):
    global running
    if key == keyboard.Key.esc:
        running = False
        return False  # stop listener
    with keys_lock:
        try:
            pressed_keys.add(key.char.lower())
        except AttributeError:
            pressed_keys.add(key)


def on_release(key):
    with keys_lock:
        try:
            pressed_keys.discard(key.char.lower())
        except AttributeError:
            pressed_keys.discard(key)


def compute_setpoint(mode):
    with keys_lock:
        keys = set(pressed_keys)

    roll = 0.0
    pitch = 0.0
    yawrate = 0.0

    if 'w' in keys:
        pitch = MAX_PITCH
    if 's' in keys:
        pitch = -MAX_PITCH
    if 'a' in keys:
        roll = -MAX_ROLL
    if 'd' in keys:
        roll = MAX_ROLL
    if 'q' in keys:
        yawrate = -MAX_YAWRATE
    if 'e' in keys:
        yawrate = MAX_YAWRATE

    if '1' in keys:
        mode = MODE_DISARM
    elif '2' in keys:
        mode = MODE_ARM
    elif '3' in keys:
        mode = MODE_HOVER

    thrust = MODE_THRUST[mode]

    if mode == MODE_HOVER:
        if keyboard.Key.space in keys:
            thrust += THRUST_STEP
        elif keyboard.Key.shift_l in keys or keyboard.Key.shift_r in keys or keyboard.Key.shift in keys:
            thrust -= THRUST_STEP

    return roll, pitch, yawrate, thrust, mode


def main():
    global running

    cflib.crtp.init_drivers()

    print('Scanning for Crazyflie...')
    available = cflib.crtp.scan_interfaces()
    if not available:
        print('No Crazyflie found! Make sure it is powered on and the Crazyradio is plugged in.')
        sys.exit(1)

    print(f'Found: {available[0][0]}')
    uri = available[0][0]

    print('Connecting...')
    cache_dir = str(Path(__file__).resolve().parent.parent / "cache")
    with SyncCrazyflie(uri, cf=Crazyflie(rw_cache=cache_dir)) as scf:
        cf = scf.cf

        # Arm and unlock motors
        cf.platform.send_arming_request(True)
        time.sleep(1.0)
        cf.commander.send_setpoint(0, 0, 0, 0)

        print()
        print('=== Keyboard Control Active ===')
        print('W/S = pitch | A/D = roll | Q/E = yaw')
        print('1 = DISARM (0) | 2 = ARM (15000) | 3 = HOVER (45000)')
        print('Space = thrust +5000 | Shift = thrust -5000')
        print('Esc = emergency stop & quit')
        print()

        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()

        mode = MODE_DISARM
        try:
            while running:
                roll, pitch, yawrate, thrust, mode = compute_setpoint(mode)
                cf.commander.send_setpoint(roll, pitch, yawrate, thrust)
                sys.stdout.write(
                    f'\rMode: {mode:7s}  Roll: {roll:+6.1f}  Pitch: {pitch:+6.1f}  '
                    f'Yaw: {yawrate:+6.1f}  Thrust: {thrust:5d}  '
                )
                sys.stdout.flush()
                time.sleep(0.02)  # 50 Hz
        finally:
            cf.commander.send_stop_setpoint()
            cf.commander.send_notify_setpoint_stop()
            time.sleep(0.1)
            listener.stop()
            print('\nDisconnected. Motors stopped.')


if __name__ == '__main__':
    main()
