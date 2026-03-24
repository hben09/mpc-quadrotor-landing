from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time
import sys
import tty
import termios
import select


# =========================
# basic settings (Drone)
# =========================
CMD_DT = 0.05   # 20 Hz command update rate

ROLL_STEP = 0.01
PITCH_STEP = 0.01
YAW_STEP = 0.01
THRUST_STEP = 0.01jksdhfadshklh

ROLL_LIMIT = 0.30
PITCH_LIMIT = 0.30
YAW_LIMIT = 0.30
THRUST_MIN = 0.0
THRUST_MAX = 100.0

DECAY_RATE = 0.5    # roll/pitch/yaw decay fraction per cycle when key released

# basic setting (Ground)
GROUND_V = -2.0



def clamp(x, xmin, xmax):
    return max(xmin, min(x, xmax))


def read_keyboard(roll, pitch, yaw, thrust):
    """
    Angled mode — roll/pitch/yaw ramp while held, decay to 0 when released.
    Thrust is persistent (only R/F/Z change it).

    Keyboard mapping:
      A / D : roll  - / +
      W / S : pitch + / -
      Q / E : yaw   + / -
      R / F : thrust + / -

      Space : reset roll/pitch/yaw to zero
      Z     : reset thrust to zero
      X     : exit
    """
    exit_flag = False
    roll_active = False
    pitch_active = False
    yaw_active = False

    while select.select([sys.stdin], [], [], 0)[0]:
        key = sys.stdin.read(1).lower()

        if key == 'a':
            roll -= ROLL_STEP
            roll_active = True
        elif key == 'd':
            roll += ROLL_STEP
            roll_active = True
        elif key == 'w':
            pitch += PITCH_STEP
            pitch_active = True
        elif key == 's':
            pitch -= PITCH_STEP
            pitch_active = True
        elif key == 'q':
            yaw += YAW_STEP
            yaw_active = True
        elif key == 'e':
            yaw -= YAW_STEP
            yaw_active = True
        elif key == 'r':
            thrust += THRUST_STEP
        elif key == 'f':
            thrust -= THRUST_STEP
        elif key == ' ':
            roll = 0.0
            pitch = 0.0
            yaw = 0.0
        elif key == 'z':
            thrust = 0.0
        elif key == 'x':
            exit_flag = True

    if not roll_active:
        roll *= (1 - DECAY_RATE)
        if abs(roll) < 0.001:
            roll = 0.0
    if not pitch_active:
        pitch *= (1 - DECAY_RATE)
        if abs(pitch) < 0.001:
            pitch = 0.0
    if not yaw_active:
        yaw *= (1 - DECAY_RATE)
        if abs(yaw) < 0.001:
            yaw = 0.0

    roll = clamp(roll, -ROLL_LIMIT, ROLL_LIMIT)
    pitch = clamp(pitch, -PITCH_LIMIT, PITCH_LIMIT)
    yaw = clamp(yaw, -YAW_LIMIT, YAW_LIMIT)
    thrust = clamp(thrust, THRUST_MIN, THRUST_MAX)

    return roll, pitch, yaw, thrust, exit_flag


def send_commands(sim, roll, pitch, yaw, thrust):
    sim.setFloatSignal('cmd_roll', float(roll))
    sim.setFloatSignal('cmd_pitch', float(pitch))
    sim.setFloatSignal('cmd_yaw', float(yaw))
    sim.setFloatSignal('cmd_thrust', float(thrust))


def reset_commands(sim):
    sim.setFloatSignal('cmd_roll', 0.0)
    sim.setFloatSignal('cmd_pitch', 0.0)
    sim.setFloatSignal('cmd_yaw', 0.0)
    sim.setFloatSignal('cmd_thrust', 0.0)


def main():
    client = RemoteAPIClient()
    sim = client.getObject('sim')

    roll = 0.0
    pitch = 0.0
    yaw = 0.0
    thrust = 0.0

    print("====================================")
    print("Manual Drone Command Test (Scheme A)")
    print("====================================")
    print("A / D : roll  - / +")
    print("W / S : pitch + / -")
    print("Q / E : yaw   + / -")
    print("R / F : thrust + / -")
    print("Space : reset roll/pitch/yaw")
    print("Z     : reset thrust to zero")
    print("X     : exit")
    print("====================================")

    sim.startSimulation()
    time.sleep(0.2)

    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    try:
        while sim.getSimulationState() != sim.simulation_stopped:
            # Drone ===========================================================
            roll, pitch, yaw, thrust, exit_flag = read_keyboard(
                roll, pitch, yaw, thrust
            )

            if exit_flag:
                break

            send_commands(sim, roll, pitch, yaw, thrust)

            print(
                f"\rroll={roll:+.3f}   "
                f"pitch={pitch:+.3f}   "
                f"yaw={yaw:+.3f}   "
                f"thrust={thrust:7.2f}",
                end=""
            )
            # Ground ==========================================================
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