"""
Main supervisor node. Adapted from RASTIC test_violation.py.
Runs at 50Hz — identical control logic to the original.
"""

import time
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Joy
from scipy.spatial.transform import Rotation as R

from crsf import (
    pwm_to_crsf, build_frame, crsf_validate_frame,
    handle_telemetry_packet,
)
from boundary import check_boundary
from pid import PIDController

import serial


# ------- Configuration -------
SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE = 115200

# Mocap topic name — change to match your rigid body name in Motive
DRONE_POSE_TOPIC = "/vrpn_mocap/AzamatDrone/pose"


class SupervisorNode(Node):
    def __init__(self):
        super().__init__('joy_to_crsf')
        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT

        self.subscription_pose = self.create_subscription(
            PoseStamped, DRONE_POSE_TOPIC, self.pose_callback, qos)
        self.subscription_joy = self.create_subscription(
            Joy, 'joy', self.joy_callback, 10)

        self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
        time.sleep(2)
        self.input_buffer = bytearray()
        threading.Thread(target=self.read_telemetry_loop, daemon=True).start()
        self.get_logger().info("Serial connection opened and ready")

        self.arm_state = 1000
        self.angle_state = 1000
        self.prev_button_4 = None
        self.prev_button_5 = None
        self.armed_flag = False
        self.last_time = time.time()

        # State
        self.x_actual = 0.0
        self.y_actual = 0.0
        self.z_actual = 0.0
        self.yaw_actual = 0.0
        self.pitch_actual = 0.0
        self.roll_actual = 0.0

        self.pid = PIDController()
        self.was_armed = False
        self.prev_cbf_bool = None
        self.return_home = False

        self.create_timer(0.02, self.control_loop)  # 50 Hz

    def read_telemetry_loop(self):
        while True:
            if self.ser.in_waiting > 0:
                self.input_buffer.extend(self.ser.read(self.ser.in_waiting))

            while len(self.input_buffer) > 2:
                expected_len = self.input_buffer[1] + 2
                if expected_len > 64 or expected_len < 4:
                    self.input_buffer = bytearray()
                    continue

                if len(self.input_buffer) >= expected_len:
                    packet = self.input_buffer[:expected_len]
                    self.input_buffer = self.input_buffer[expected_len:]
                    if crsf_validate_frame(packet):
                        handle_telemetry_packet(packet[2], packet)
                    else:
                        print(f"crc error: {' '.join(hex(b) for b in packet)}")
                else:
                    break
            time.sleep(0.001)

    def pose_callback(self, msg):
        self.x_actual = msg.pose.position.x
        self.y_actual = msg.pose.position.y
        self.z_actual = msg.pose.position.z
        q = msg.pose.orientation
        r = R.from_quat([q.x, q.y, q.z, q.w])
        self.roll_actual, self.pitch_actual, self.yaw_actual = r.as_euler('xyz', degrees=False)

    def joy_callback(self, msg: Joy):
        current_button_4 = msg.buttons[4]
        current_button_5 = msg.buttons[5]

        if self.prev_button_4 is None:
            self.prev_button_4 = current_button_4
            return
        if self.prev_button_5 is None:
            self.prev_button_5 = current_button_5
            return

        if current_button_4 == 1 and self.prev_button_4 == 0:
            self.arm_state = 2000 if self.arm_state == 1000 else 1000
            print(f"Toggled arm: {self.arm_state}")
        if current_button_5 == 1 and self.prev_button_5 == 0:
            self.angle_state = 1500 if self.angle_state == 1000 else 1000
            print(f"Toggled angle: {self.angle_state}")

        self.prev_button_4 = current_button_4
        self.prev_button_5 = current_button_5

    def control_loop(self):
        current_position = [self.x_actual, self.y_actual, self.z_actual]
        roll = pwm_to_crsf(1500)
        pitch = pwm_to_crsf(1500)
        yaw = pwm_to_crsf(1500)
        throttle = pwm_to_crsf(1000)

        if self.arm_state == 2000:
            if not self.armed_flag:
                throttle = pwm_to_crsf(1000)
                self.armed_flag = True
                self.pid.update_state(self.x_actual, self.y_actual, self.z_actual,
                                      self.roll_actual, self.pitch_actual, self.yaw_actual)
                self.pid.reset()
                self.was_armed = True
                print("Armed, holding throttle min & PID reset")
            else:
                cbf_violation = check_boundary(current_position)
                current_cbf_bool = cbf_violation
                if self.prev_cbf_bool is None:
                    self.prev_cbf_bool = current_cbf_bool
                    return
                if current_cbf_bool is True and self.prev_cbf_bool is False:
                    self.return_home = True
                    print("home", self.x_actual, self.y_actual, self.z_actual)
                if self.return_home:
                    target = [0, 0.6, 0]
                else:
                    target = [3.0, 0.6, 0.0]

                self.prev_cbf_bool = current_cbf_bool
                self.pid.update_state(self.x_actual, self.y_actual, self.z_actual,
                                      self.roll_actual, self.pitch_actual, self.yaw_actual)
                roll, pitch, throttle, yaw = self.pid.compute(target)
        else:
            roll = 1500
            pitch = 1500
            yaw = 1500
            throttle = 1000
            if self.was_armed:
                self.pid.reset()
                self.was_armed = False
            self.armed_flag = False

        aux = [pwm_to_crsf(1000)] * 10
        channels = [
            pwm_to_crsf(roll), pwm_to_crsf(pitch),
            pwm_to_crsf(throttle), pwm_to_crsf(yaw),
            pwm_to_crsf(self.arm_state), pwm_to_crsf(self.angle_state),
        ] + aux
        frame = build_frame(channels)
        self.ser.write(frame)


def main(args=None):
    rclpy.init(args=args)
    node = SupervisorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
