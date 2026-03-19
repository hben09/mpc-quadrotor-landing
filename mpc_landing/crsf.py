"""
CRSF protocol utilities for ELRS communication.
Adapted from RASTIC safeguard system (MIT License, CRSF Working Group).
"""

from enum import IntEnum


CRSF_SYNC = 0xC8
PACKET_TYPE = 0x16


class PacketsTypes(IntEnum):
    GPS = 0x02
    VARIO = 0x07
    BATTERY_SENSOR = 0x08
    BARO_ALT = 0x09
    HEARTBEAT = 0x0B
    VIDEO_TRANSMITTER = 0x0F
    LINK_STATISTICS = 0x14
    RC_CHANNELS_PACKED = 0x16
    ATTITUDE = 0x1E
    FLIGHT_MODE = 0x21
    DEVICE_INFO = 0x29
    CONFIG_READ = 0x2C
    CONFIG_WRITE = 0x2D
    RADIO_ID = 0x3A


def pwm_to_crsf(pwm_us: int) -> int:
    pwm_us = max(1000, min(2000, pwm_us))
    return int((pwm_us - 1000) * (1792 - 191) / (2000 - 1000) + 191)


def crc8_dvb_s2(crc, a):
    crc ^= a
    for _ in range(8):
        if crc & 0x80:
            crc = (crc << 1) ^ 0xD5
        else:
            crc <<= 1
    return crc & 0xFF


def crc8_data(data):
    crc = 0
    for b in data:
        crc = crc8_dvb_s2(crc, b)
    return crc


def crsf_validate_frame(frame) -> bool:
    return crc8_data(frame[2:-1]) == frame[-1]


def signed_byte(b):
    return b - 256 if b >= 128 else b


def pack_crsf_channels(channels):
    b = bytearray(22)
    bits = 0
    bitpos = 0
    bytepos = 0
    for ch in channels:
        bits |= (ch & 0x7FF) << bitpos
        bitpos += 11
        while bitpos >= 8:
            b[bytepos] = bits & 0xFF
            bits >>= 8
            bitpos -= 8
            bytepos += 1
    return b


def build_frame(channels):
    payload = bytearray()
    payload.append(PACKET_TYPE)
    payload += pack_crsf_channels(channels)
    frame = bytearray()
    frame.append(CRSF_SYNC)
    frame.append(len(payload) + 1)
    frame += payload
    frame.append(crc8_data(payload))
    return frame


def handle_telemetry_packet(ptype, data):
    if ptype == PacketsTypes.FLIGHT_MODE:
        packet = ''.join(map(chr, data[3:-2]))
        print(f"Flight Mode: {packet}")
    elif ptype == PacketsTypes.BATTERY_SENSOR:
        vbat = int.from_bytes(data[3:5], byteorder='big', signed=True) / 10.0
        curr = int.from_bytes(data[5:7], byteorder='big', signed=True) / 10.0
        pct = data[10]
        print(f"Battery: {vbat:0.2f}V {curr:0.1f}A {pct}%")
