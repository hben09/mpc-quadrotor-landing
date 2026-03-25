"""
Manual keyboard control via CoppeliaSim ZMQ RemoteAPI.

Uses pynput for proper keydown/keyup detection — no terminal key repeat
issues. Sends PWM values (1000-2000) matching the Betaflight angle mode
interface used by the Lua flight controller.

Keyboard mapping:
    A / D : roll  - / +
    W / S : pitch + / -
    Q / E : yaw   + / -
    R / F : thrust + / -

    Space : center roll/pitch/yaw (1500)
    Z     : zero thrust (1000)
    Esc   : exit
"""

from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from pynput import keyboard
import time
import threading


# =========================
# PWM settings
# =========================
CMD_DT = 0.05          # 20 Hz command update rate

RAMP_RATE = 15         # PWM change per cycle while key held (15 * 20Hz = 300 PWM/s)
THRUST_RATE = 5        # finer throttle ramp
DECAY_RATE = 0.15      # roll/pitch/yaw decay toward center when key released

PWM_CENTER = 1500      # neutral for roll/pitch/yaw
PWM_MIN = 1000
PWM_MAX = 2000
THRUST_IDLE = 1000

# Ground platform speed
GROUND_V = -2.0


def clamp(x, xmin, xmax):
    return max(xmin, min(x, xmax))


def decay_toward_center(val, center, rate):
    """Exponentially decay a PWM value toward center."""
    diff = val - center
    diff *= (1 - rate)
    if abs(diff) < 1:
        return center
    return center + diff


class KeyState:
    """Thread-safe tracking of currently pressed keys via pynput."""

    def __init__(self):
        self._pressed = set()
        self._lock = threading.Lock()
        self.exit_flag = False

    def on_press(self, key):
        try:
            k = key.char
        except AttributeError:
            k = key
        with self._lock:
            self._pressed.add(k)
        if key == keyboard.Key.esc:
            self.exit_flag = True

    def on_release(self, key):
        try:
            k = key.char
        except AttributeError:
            k = key
        with self._lock:
            self._pressed.discard(k)

    def is_pressed(self, char):
        with self._lock:
            return char in self._pressed

    def is_any_pressed(self, *chars):
        with self._lock:
            return any(c in self._pressed for c in chars)


def send_commands(sim, roll, pitch, yaw, thrust):
    sim.setFloatSignal('cmd_roll', float(roll))
    sim.setFloatSignal('cmd_pitch', float(pitch))
    sim.setFloatSignal('cmd_yaw', float(yaw))
    sim.setFloatSignal('cmd_thrust', float(thrust))


def reset_commands(sim):
    sim.setFloatSignal('cmd_roll', float(PWM_CENTER))
    sim.setFloatSignal('cmd_pitch', float(PWM_CENTER))
    sim.setFloatSignal('cmd_yaw', float(PWM_CENTER))
    sim.setFloatSignal('cmd_thrust', float(THRUST_IDLE))


def main():
    client = RemoteAPIClient()
    sim = client.getObject('sim')

    roll = PWM_CENTER
    pitch = PWM_CENTER
    yaw = PWM_CENTER
    thrust = THRUST_IDLE

    keys = KeyState()
    listener = keyboard.Listener(on_press=keys.on_press, on_release=keys.on_release)
    listener.start()

    print("==========================================")
    print("Manual Drone Control (Betaflight PWM Mode)")
    print("==========================================")
    print("A / D : roll  - / +")
    print("W / S : pitch + / -")
    print("Q / E : yaw   + / -")
    print("R / F : thrust + / -")
    print("Space : center roll/pitch/yaw (1500)")
    print("Z     : zero thrust (1000)")
    print("Esc   : exit")
    print(f"PWM range: {PWM_MIN}-{PWM_MAX}")
    print("==========================================")

    sim.startSimulation()
    time.sleep(0.2)

    try:
        while sim.getSimulationState() != sim.simulation_stopped and not keys.exit_flag:
            # Roll: A/D
            if keys.is_pressed('a'):
                roll -= RAMP_RATE
            elif keys.is_pressed('d'):
                roll += RAMP_RATE
            else:
                roll = decay_toward_center(roll, PWM_CENTER, DECAY_RATE)

            # Pitch: W/S
            if keys.is_pressed('w'):
                pitch += RAMP_RATE
            elif keys.is_pressed('s'):
                pitch -= RAMP_RATE
            else:
                pitch = decay_toward_center(pitch, PWM_CENTER, DECAY_RATE)

            # Yaw: Q/E
            if keys.is_pressed('q'):
                yaw += RAMP_RATE
            elif keys.is_pressed('e'):
                yaw -= RAMP_RATE
            else:
                yaw = decay_toward_center(yaw, PWM_CENTER, DECAY_RATE)

            # Thrust: R/F (persistent, no decay)
            if keys.is_pressed('r'):
                thrust += THRUST_RATE
            elif keys.is_pressed('f'):
                thrust -= THRUST_RATE

            # Space: center roll/pitch/yaw
            if keys.is_pressed(' '):
                roll = PWM_CENTER
                pitch = PWM_CENTER
                yaw = PWM_CENTER

            # Z: zero thrust
            if keys.is_pressed('z'):
                thrust = THRUST_IDLE

            # Clamp
            roll = clamp(roll, PWM_MIN, PWM_MAX)
            pitch = clamp(pitch, PWM_MIN, PWM_MAX)
            yaw = clamp(yaw, PWM_MIN, PWM_MAX)
            thrust = clamp(thrust, PWM_MIN, PWM_MAX)

            send_commands(sim, roll, pitch, yaw, thrust)

            print(
                f"\rroll={roll:7.1f}   "
                f"pitch={pitch:7.1f}   "
                f"yaw={yaw:7.1f}   "
                f"thrust={thrust:7.1f}",
                end=""
            )

            # Ground platform
            sim.setFloatSignal('ground_v', GROUND_V)

            time.sleep(CMD_DT)

    finally:
        print("\nStopping simulation...")
        listener.stop()
        reset_commands(sim)
        time.sleep(0.1)
        sim.stopSimulation()


if __name__ == "__main__":
    main()
