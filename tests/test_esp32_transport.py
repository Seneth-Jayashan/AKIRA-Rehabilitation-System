"""Automated mock-transport and packet validation tests for Live ESP32 Dual-IMU testing.

NOTE: All tests below use software mock transports and synthetic test streams.
No physical ESP32 microcontrollers are claimed to be attached during these automated unit tests.
"""

from __future__ import annotations

import struct
import time
import pytest

from src.inference.esp32_live_validator import (
    ESP32LiveStreamValidator,
    ConnectionState,
    IMUStatus,
    CANONICAL_CHANNELS,
    MAGIC_HEADER,
)


def make_valid_binary_packet(seq: int, t_ms: int, status_flags: int = 0) -> bytes:
    """Helper to build a valid 64-byte Little-Endian binary packet."""
    # 12 floats: RUA (Ax=0, Ay=9.81, Az=0, Gx=0.1, Gy=0.2, Gz=0.3)
    #            RLA (Ax=0, Ay=9.81, Az=0, Gx=0.2, Gy=0.1, Gz=0.4)
    floats = [0.0, 9.81, 0.0, 0.1, 0.2, 0.3, 0.0, 9.81, 0.0, 0.2, 0.1, 0.4]
    header = struct.pack("<III", MAGIC_HEADER, t_ms, seq)
    payload = struct.pack("<12f", *floats)
    flags = struct.pack("<H", status_flags)
    raw_62 = header + payload + flags
    checksum = sum(raw_62) & 0xFFFF
    trailer = struct.pack("<H", checksum)
    return raw_62 + trailer


def test_no_esp32_connected_initial_state():
    """Test 1: Initial state is Disconnected and never falsely claims Connected."""
    validator = ESP32LiveStreamValidator()
    assert validator.current_state == ConnectionState.DISCONNECTED
    rep = validator.get_status_report()
    assert rep["is_verified"] is False
    assert rep["state"] == "Disconnected"


def test_device_selected_awaiting_handshake():
    """Test 2: Device selected transitions to Searching / Awaiting Response, never verified prematurely."""
    validator = ESP32LiveStreamValidator()
    validator.start_connecting()
    assert validator.current_state == ConnectionState.SEARCHING
    assert validator.get_status_report()["is_verified"] is False

    validator.port_opened_awaiting_handshake()
    assert validator.current_state == ConnectionState.AWAITING_RESPONSE
    assert validator.get_status_report()["is_verified"] is False

    # Timeout occurs when no packet arrives
    validator.handle_connection_failure("Handshake timeout")
    assert validator.current_state == ConnectionState.CONNECTION_FAILED
    assert validator.get_status_report()["is_verified"] is False


def test_valid_binary_packet_both_imus():
    """Test 3: Valid 64-byte binary packet transitions state to Connected — Verified."""
    validator = ESP32LiveStreamValidator()
    validator.port_opened_awaiting_handshake()

    pkt_bytes = make_valid_binary_packet(seq=1, t_ms=10)
    res = validator.validate_binary_packet(pkt_bytes)

    assert res.is_valid is True
    assert res.state == ConnectionState.CONNECTED_VERIFIED
    assert res.both_imus_active is True
    assert res.rua_status == IMUStatus.HEALTHY
    assert res.rla_status == IMUStatus.HEALTHY
    assert len(res.channels) == 12
    assert set(res.channels.keys()) == set(CANONICAL_CHANNELS)
    assert res.channels["RUA_Ay"] == 9.81
    assert res.channels["RLA_Ay"] == 9.81


def test_single_imu_missing_warning():
    """Test 4: Connected ESP32 with only one active IMU displays single-IMU warning."""
    validator = ESP32LiveStreamValidator()

    # RLA is zeroed / flagged with fault bit 0x02
    floats = [0.0, 9.81, 0.0, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    header = struct.pack("<III", MAGIC_HEADER, 100, 1)
    payload = struct.pack("<12f", *floats)
    flags = struct.pack("<H", 0x02)  # IMU2 fault flag
    raw_60 = header + payload + flags
    csum = sum(raw_60) & 0xFFFF
    pkt = raw_60 + struct.pack("<H", csum)

    res = validator.validate_binary_packet(pkt)
    assert res.is_valid is True
    assert res.both_imus_active is False
    assert res.rua_status == IMUStatus.HEALTHY
    assert res.rla_status == IMUStatus.MISSING
    assert res.warning_message is not None
    assert "Single-IMU Warning" in res.warning_message
    assert "Forearm IMU (RLA)" in res.warning_message


def test_malformed_packets_rejected():
    """Test 5: Malformed magic headers and wrong lengths are rejected."""
    validator = ESP32LiveStreamValidator()

    # Too short
    short_bytes = b"12345"
    res1 = validator.validate_binary_packet(short_bytes)
    assert res1.is_valid is False
    assert res1.state == ConnectionState.INVALID_SENSOR_DATA

    # Bad magic header
    bad_magic = struct.pack("<III", 0xDEADBEEF, 100, 1) + b"\x00" * 52
    res2 = validator.validate_binary_packet(bad_magic)
    assert res2.is_valid is False
    assert "Magic header mismatch" in res2.error_message


def test_out_of_bounds_sensor_data_rejected():
    """Test 6: Extreme sensor anomalies (> 150 m/s^2 or > 35 rad/s) are rejected."""
    validator = ESP32LiveStreamValidator()

    # Acceleration 999.0 m/s^2
    floats = [999.0, 9.81, 0.0, 0.1, 0.2, 0.3, 0.0, 9.81, 0.0, 0.2, 0.1, 0.4]
    header = struct.pack("<III", MAGIC_HEADER, 100, 1)
    payload = struct.pack("<12f", *floats)
    flags = struct.pack("<H", 0)
    raw_60 = header + payload + flags
    csum = sum(raw_60) & 0xFFFF
    pkt = raw_60 + struct.pack("<H", csum)

    res = validator.validate_binary_packet(pkt)
    assert res.is_valid is False
    assert res.state == ConnectionState.INVALID_SENSOR_DATA
    assert "exceeded safety threshold" in res.error_message


def test_device_disconnect_during_streaming_timeout():
    """Test 7: Stream stall beyond keepalive threshold transitions to Device Disconnected."""
    validator = ESP32LiveStreamValidator(disconnect_timeout_sec=0.5)
    t0 = 1000.0
    pkt = make_valid_binary_packet(seq=1, t_ms=10)
    validator.validate_binary_packet(pkt, receive_time=t0)
    assert validator.current_state == ConnectionState.CONNECTED_VERIFIED

    # Check keepalive 0.2s later (still alive)
    ok1, _ = validator.check_keepalive(current_time=t0 + 0.2)
    assert ok1 is True
    assert validator.current_state == ConnectionState.CONNECTED_VERIFIED

    # Check keepalive 0.8s later (stalled > 0.5s)
    ok2, err = validator.check_keepalive(current_time=t0 + 0.8)
    assert ok2 is False
    assert validator.current_state == ConnectionState.DEVICE_DISCONNECTED
    assert "Stream stalled" in err


def test_reconnection_after_failure():
    """Test 8: System handles failure and successfully reconnects on valid packet arrival."""
    validator = ESP32LiveStreamValidator()
    validator.handle_connection_failure("Initial port busy")
    assert validator.current_state == ConnectionState.CONNECTION_FAILED

    # Start new connection attempt
    validator.start_connecting()
    assert validator.current_state == ConnectionState.SEARCHING

    # Valid packet arrives
    pkt = make_valid_binary_packet(seq=1, t_ms=10)
    res = validator.validate_binary_packet(pkt)
    assert res.is_valid is True
    assert validator.current_state == ConnectionState.CONNECTED_VERIFIED


def test_json_and_csv_streaming_validation():
    """Test 9: JSON and CSV streaming lines parse and validate accurately."""
    validator = ESP32LiveStreamValidator()

    # Valid JSON
    json_packet = (
        '{"t_ms": 500, "seq": 10, "rua": [0.0, 9.81, 0.0, 0.1, 0.0, 0.0], '
        '"rla": [0.0, 9.81, 0.0, 0.2, 0.0, 0.0], "status": 0}'
    )
    res_json = validator.validate_json_packet(json_packet)
    assert res_json.is_valid is True
    assert res_json.both_imus_active is True
    assert res_json.channels["RUA_Ay"] == 9.81

    # Valid CSV Line
    csv_line = "510,11,0.0,9.81,0.0,0.1,0.0,0.0,0.0,9.81,0.0,0.2,0.0,0.0"
    res_csv = validator.validate_csv_line(csv_line)
    assert res_csv.is_valid is True
    assert res_csv.channels["RLA_Ay"] == 9.81


def test_simulation_mode_is_explicitly_labeled():
    """Test 10: Simulation mode is explicitly separated and never claims hardware verified."""
    validator = ESP32LiveStreamValidator()
    validator.is_simulation = True
    validator.current_state = ConnectionState.SIMULATION_REPLAY

    report = validator.get_status_report()
    assert report["is_simulation"] is True
    assert report["is_verified"] is False
    assert report["state"] == "Simulation Replay (Offline)"
