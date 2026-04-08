import logging
import sys
import time
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
THRUST_STEP = 500

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


def compute_setpoint(thrust):
    with keys_lock:
        keys = set(pressed_keys)

    roll = 0.0
    pitch = 0.0
    yawrate = 0.0

    if 'w' in keys:
        pitch = -MAX_PITCH   # negative pitch = forward
    if 's' in keys:
        pitch = MAX_PITCH
    if 'a' in keys:
        roll = -MAX_ROLL     # negative roll = left
    if 'd' in keys:
        roll = MAX_ROLL
    if 'q' in keys:
        yawrate = -MAX_YAWRATE
    if 'e' in keys:
        yawrate = MAX_YAWRATE

    if keyboard.Key.space in keys:
        thrust = min(thrust + THRUST_STEP, MAX_THRUST)
    elif keyboard.Key.shift_l in keys or keyboard.Key.shift_r in keys or keyboard.Key.shift in keys:
        thrust = max(thrust - THRUST_STEP, 0)
    else:
        # Gentle decay when no thrust key held
        thrust = max(thrust - 100, 0)

    return roll, pitch, yawrate, int(thrust)


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
    with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')) as scf:
        cf = scf.cf

        # Arm and unlock motors
        cf.platform.send_arming_request(True)
        time.sleep(1.0)
        cf.commander.send_setpoint(0, 0, 0, 0)

        print()
        print('=== Keyboard Control Active ===')
        print('W/S = pitch | A/D = roll | Q/E = yaw')
        print('Space = thrust up | Shift = thrust down')
        print('Esc = emergency stop & quit')
        print()

        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()

        thrust = 0
        try:
            while running:
                roll, pitch, yawrate, thrust = compute_setpoint(thrust)
                cf.commander.send_setpoint(roll, pitch, yawrate, thrust)
                sys.stdout.write(
                    f'\rRoll: {roll:+6.1f}  Pitch: {pitch:+6.1f}  '
                    f'Yaw: {yawrate:+6.1f}  Thrust: {thrust:5d}  '
                )
                sys.stdout.flush()
                time.sleep(0.02)  # 50 Hz
        finally:
            cf.commander.send_stop_setpoint()
            listener.stop()
            print('\nDisconnected. Motors stopped.')


if __name__ == '__main__':
    main()
