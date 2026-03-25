"""
Manual keyboard control via CoppeliaSim ZMQ RemoteAPI.

Sends PWM values (1000-2000) matching the Betaflight angle mode interface
used by the Lua flight controller.

Keyboard mapping:
    A / D : roll  - / +
    W / S : pitch + / -
    Q / E : yaw   + / -
    R / F : thrust + / -

    Space : center roll/pitch/yaw (1500)
    Z     : zero thrust (1000)
    X     : exit
"""

from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time
import sys
import tty
import termios
import select


# =========================
# PWM settings
# =========================
CMD_DT = 0.05   # 20 Hz command update rate

PWM_STEP = 25          # PWM increment per keypress (roll/pitch/yaw)
THRUST_STEP = 5        # finer throttle control near hover

PWM_CENTER = 1500      # neutral for roll/pitch/yaw
PWM_MIN = 1000
PWM_MAX = 2000
THRUST_IDLE = 1000

DECAY_RATE = 0.15      # roll/pitch/yaw decay toward center when key released

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


def read_keyboard(roll, pitch, yaw, thrust):
    """Read keyboard and update PWM commands.

    Roll/pitch/yaw ramp while held, decay to 1500 when released.
    Thrust is persistent (only R/F/Z change it).
    """
    exit_flag = False
    roll_active = False
    pitch_active = False
    yaw_active = False

    while select.select([sys.stdin], [], [], 0)[0]:
        key = sys.stdin.read(1).lower()

        if key == 'a':
            roll -= PWM_STEP
            roll_active = True
        elif key == 'd':
            roll += PWM_STEP
            roll_active = True
        elif key == 'w':
            pitch += PWM_STEP
            pitch_active = True
        elif key == 's':
            pitch -= PWM_STEP
            pitch_active = True
        elif key == 'q':
            yaw += PWM_STEP
            yaw_active = True
        elif key == 'e':
            yaw -= PWM_STEP
            yaw_active = True
        elif key == 'r':
            thrust += THRUST_STEP
        elif key == 'f':
            thrust -= THRUST_STEP
        elif key == ' ':
            roll = PWM_CENTER
            pitch = PWM_CENTER
            yaw = PWM_CENTER
        elif key == 'z':
            thrust = THRUST_IDLE
        elif key == 'x':
            exit_flag = True

    # Decay roll/pitch/yaw toward center when not actively pressed
    if not roll_active:
        roll = decay_toward_center(roll, PWM_CENTER, DECAY_RATE)
    if not pitch_active:
        pitch = decay_toward_center(pitch, PWM_CENTER, DECAY_RATE)
    if not yaw_active:
        yaw = decay_toward_center(yaw, PWM_CENTER, DECAY_RATE)

    roll = clamp(roll, PWM_MIN, PWM_MAX)
    pitch = clamp(pitch, PWM_MIN, PWM_MAX)
    yaw = clamp(yaw, PWM_MIN, PWM_MAX)
    thrust = clamp(thrust, PWM_MIN, PWM_MAX)

    return roll, pitch, yaw, thrust, exit_flag


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

    print("==========================================")
    print("Manual Drone Control (Betaflight PWM Mode)")
    print("==========================================")
    print("A / D : roll  - / +")
    print("W / S : pitch + / -")
    print("Q / E : yaw   + / -")
    print("R / F : thrust + / -")
    print("Space : center roll/pitch/yaw (1500)")
    print("Z     : zero thrust (1000)")
    print("X     : exit")
    print(f"PWM range: {PWM_MIN}-{PWM_MAX}, step: {PWM_STEP}")
    print("==========================================")

    sim.startSimulation()
    time.sleep(0.2)

    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    try:
        while sim.getSimulationState() != sim.simulation_stopped:
            roll, pitch, yaw, thrust, exit_flag = read_keyboard(
                roll, pitch, yaw, thrust
            )

            if exit_flag:
                break

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
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        print("\nStopping simulation...")
        reset_commands(sim)
        time.sleep(0.1)
        sim.stopSimulation()


if __name__ == "__main__":
    main()
