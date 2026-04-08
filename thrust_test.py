import logging
import sys
import time
from threading import Lock

from pynput import keyboard

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils import uri_helper

MAX_THRUST = 60000
COARSE_STEP = 500
FINE_STEP = 100

logging.basicConfig(level=logging.ERROR)

thrust = 0
thrust_lock = Lock()
running = True


def on_press(key):
    global thrust, running

    if key == keyboard.Key.esc:
        with thrust_lock:
            thrust = 0
        running = False
        return False

    with thrust_lock:
        if key == keyboard.Key.up:
            thrust = min(thrust + COARSE_STEP, MAX_THRUST)
        elif key == keyboard.Key.down:
            thrust = max(thrust - COARSE_STEP, 0)
        elif hasattr(key, 'char') and key.char == 'w':
            thrust = min(thrust + FINE_STEP, MAX_THRUST)
        elif hasattr(key, 'char') and key.char == 's':
            thrust = max(thrust - FINE_STEP, 0)


def main():
    global running

    cflib.crtp.init_drivers()

    print('Scanning for Crazyflie...')
    available = cflib.crtp.scan_interfaces()
    if not available:
        print('No Crazyflie found!')
        sys.exit(1)

    uri = available[0][0]
    print(f'Found: {uri}')
    print('Connecting...')

    with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')) as scf:
        cf = scf.cf

        cf.platform.send_arming_request(True)
        time.sleep(1.0)
        cf.commander.send_setpoint(0, 0, 0, 0)

        print()
        print('=== Thrust Test ===')
        print('Up/Down = +/- 500 (coarse)')
        print('W/S     = +/- 100 (fine)')
        print('Esc     = EMERGENCY STOP & quit')
        print()

        listener = keyboard.Listener(on_press=on_press)
        listener.start()

        try:
            while running:
                with thrust_lock:
                    t = thrust
                cf.commander.send_setpoint(0, 0, 0, t)
                sys.stdout.write(f'\rThrust: {t:5d} / {MAX_THRUST}    ')
                sys.stdout.flush()
                time.sleep(0.02)
        finally:
            cf.commander.send_stop_setpoint()
            cf.commander.send_notify_setpoint_stop()
            time.sleep(0.1)
            listener.stop()
            print('\nMotors stopped.')


if __name__ == '__main__':
    main()
