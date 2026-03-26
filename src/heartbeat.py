"""
ELRS heartbeat: sends neutral CRSF frames so the TX module
recognises the PC as a connected handset.

Usage: python heartbeat.py [serial_port]
"""

import glob
import sys
import time
import threading

import serial

from crsf import (
    pwm_to_crsf, build_frame, crsf_validate_frame,
    handle_telemetry_packet,
)

BAUD_RATE = 420000
RATE_HZ = 50


def auto_detect_port():
    ports = sorted(glob.glob("/dev/cu.usbserial-*"))
    if len(ports) == 1:
        return ports[0]
    if len(ports) > 1:
        print("Multiple USB-serial devices found:")
        for p in ports:
            print(f"  {p}")
        sys.exit("Specify the port: python heartbeat.py <port>")
    sys.exit("No USB-serial device found. Is the ELRS TX module plugged in?")


def telemetry_reader(ser):
    buf = bytearray()
    while True:
        if ser.in_waiting > 0:
            buf.extend(ser.read(ser.in_waiting))

        while len(buf) > 2:
            expected_len = buf[1] + 2
            if expected_len > 64 or expected_len < 4:
                buf.clear()
                continue
            if len(buf) >= expected_len:
                packet = buf[:expected_len]
                buf = buf[expected_len:]
                if crsf_validate_frame(packet):
                    handle_telemetry_packet(packet[2], packet)
            else:
                break
        time.sleep(0.001)


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else auto_detect_port()
    print(f"Opening {port} at {BAUD_RATE} baud ...")
    ser = serial.Serial(port, BAUD_RATE)
    time.sleep(2)
    print("Connected. Sending heartbeat frames at 50 Hz. Ctrl+C to stop.")

    threading.Thread(target=telemetry_reader, args=(ser,), daemon=True).start()

    # Neutral / disarmed channels
    neutral = [
        pwm_to_crsf(1500),  # roll
        pwm_to_crsf(1500),  # pitch
        pwm_to_crsf(1000),  # throttle
        pwm_to_crsf(1500),  # yaw
        pwm_to_crsf(1000),  # arm  (disarmed)
        pwm_to_crsf(1000),  # angle mode (off)
    ] + [pwm_to_crsf(1000)] * 10  # aux

    frame = build_frame(neutral)
    period = 1.0 / RATE_HZ

    try:
        while True:
            ser.write(frame)
            time.sleep(period)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        ser.close()


if __name__ == "__main__":
    main()
