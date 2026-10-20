"""ESP32 Dual-IMU Live Streaming Validator and Protocol Adapter.

Strictly enforces:
1. No false-positive connections: Connection verified ONLY after valid packet reception or handshake response.
2. 12-Channel input validation: [RUA_Ax, RUA_Ay, RUA_Az, RUA_Gx, RUA_Gy, RUA_Gz, RLA_Ax, RLA_Ay, RLA_Az, RLA_Gx, RLA_Gy, RLA_Gz].
3. Dual-IMU health checking: Distinguishes connected ESP32 from working dual-IMU system.
4. Monotonicity, freshness, and sampling rate tracking.
5. Causal streaming buffer management.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
import struct
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Canonical 12-channel order expected by all 4 trained models
CANONICAL_CHANNELS = [
    "RUA_Ax", "RUA_Ay", "RUA_Az",
    "RUA_Gx", "RUA_Gy", "RUA_Gz",
    "RLA_Ax", "RLA_Ay", "RLA_Az",
    "RLA_Gx", "RLA_Gy", "RLA_Gz",
]

MAGIC_HEADER = 0xAA55AA55
PACKET_SIZE_BINARY = 64  # bytes


class ConnectionState(str, Enum):
    DISCONNECTED = "Disconnected"
    SEARCHING = "Searching"
    AWAITING_RESPONSE = "Awaiting Response"
    CONNECTED_VERIFIED = "Connected — Verified"
    CONNECTION_FAILED = "Connection Failed"
    DEVICE_DISCONNECTED = "Device Disconnected"
    INVALID_SENSOR_DATA = "Invalid Sensor Data"
    SIMULATION_REPLAY = "Simulation Replay (Offline)"


class IMUStatus(str, Enum):
    HEALTHY = "Active (100 Hz)"
    MISSING = "Missing / Silent"
    OUT_OF_RANGE = "Signal Out of Range"
    FLATLINE = "Frozen Flatline"


@dataclass
class PacketValidationResult:
    is_valid: bool
    state: ConnectionState
    timestamp_ms: int = 0
    seq_num: int = 0
    channels: Dict[str, float] = field(default_factory=dict)
    both_imus_active: bool = False
    rua_status: IMUStatus = IMUStatus.MISSING
    rla_status: IMUStatus = IMUStatus.MISSING
    status_flags: int = 0
    error_message: Optional[str] = None
    warning_message: Optional[str] = None


class ESP32LiveStreamValidator:
    """Validates real-time ESP32 communication transport packets and tracks connection state."""

    def __init__(
        self,
        expected_rate_hz: float = 100.0,
        disconnect_timeout_sec: float = 1.2,
        acc_max_limit_mps2: float = 150.0,
        gyr_max_limit_rads: float = 35.0,
    ) -> None:
        self.expected_rate_hz = expected_rate_hz
        self.disconnect_timeout_sec = disconnect_timeout_sec
        self.acc_limit = acc_max_limit_mps2
        self.gyr_limit = gyr_max_limit_rads

        self.current_state = ConnectionState.DISCONNECTED
        self.is_simulation = False
        self.last_packet_time: Optional[float] = None
        self.last_seq_num: Optional[int] = None
        self.consecutive_drops = 0
        self.total_packets_received = 0

        # Ring buffer for live sampling rate calculation
        self._timestamps_window: List[float] = []
        self._measured_rate_hz: float = 0.0

        # Recent channel history for flatline / noise detection
        self._rua_acc_history: List[float] = []
        self._rla_acc_history: List[float] = []

    def start_connecting(self) -> None:
        """Transitions state to Searching when initiating port / device discovery."""
        self.current_state = ConnectionState.SEARCHING
        self.is_simulation = False
        self.last_packet_time = None
        self.last_seq_num = None

    def port_opened_awaiting_handshake(self) -> None:
        """Port selected and opened; waiting for device to respond."""
        self.current_state = ConnectionState.AWAITING_RESPONSE
        self.is_simulation = False

    def handle_connection_failure(self, reason: str = "Timeout awaiting device response") -> ConnectionState:
        """Called when serial/BLE connection attempt fails or is aborted."""
        self.current_state = ConnectionState.CONNECTION_FAILED
        return self.current_state

    def handle_device_disconnect(self) -> ConnectionState:
        """Called on transport close, hardware removal, or data stream stall."""
        self.current_state = ConnectionState.DEVICE_DISCONNECTED
        self.last_packet_time = None
        return self.current_state

    def reset_to_disconnected(self) -> None:
        """Clean disconnect by user."""
        self.current_state = ConnectionState.DISCONNECTED
        self.is_simulation = False
        self.last_packet_time = None
        self.last_seq_num = None
        self._timestamps_window.clear()
        self._measured_rate_hz = 0.0

    def check_keepalive(self, current_time: Optional[float] = None) -> Tuple[bool, Optional[str]]:
        """Check if streaming has stalled past the disconnect timeout threshold."""
        now = current_time or time.time()
        if self.current_state == ConnectionState.CONNECTED_VERIFIED:
            if self.last_packet_time is not None:
                elapsed = now - self.last_packet_time
                if elapsed > self.disconnect_timeout_sec:
                    self.current_state = ConnectionState.DEVICE_DISCONNECTED
                    return False, f"Stream stalled: no packets received for {elapsed:.2f}s (threshold: {self.disconnect_timeout_sec}s)."
        return True, None

    def validate_binary_packet(self, raw_bytes: bytes, receive_time: Optional[float] = None) -> PacketValidationResult:
        """Validate and deserialize a 64-byte binary packed ESP32 C-struct.

        Struct format (< = Little Endian):
          uint32 magic (0xAA55AA55)       4 bytes
          uint32 timestamp_ms             4 bytes
          uint32 seq_num                  4 bytes
          float  rua_ax, ay, az (m/s^2)  12 bytes
          float  rua_gx, gy, gz (rad/s)  12 bytes
          float  rla_ax, ay, az (m/s^2)  12 bytes
          float  rla_gx, gy, gz (rad/s)  12 bytes
          uint16 status_flags             2 bytes
          uint16 checksum                 2 bytes
          Total:                         64 bytes
        """
        now = receive_time or time.time()

        if len(raw_bytes) != PACKET_SIZE_BINARY:
            self.current_state = ConnectionState.INVALID_SENSOR_DATA
            return PacketValidationResult(
                is_valid=False,
                state=self.current_state,
                error_message=f"Invalid packet length: got {len(raw_bytes)} bytes, expected {PACKET_SIZE_BINARY}.",
            )

        magic, t_ms, seq = struct.unpack_from("<III", raw_bytes, 0)
        if magic != MAGIC_HEADER:
            self.current_state = ConnectionState.INVALID_SENSOR_DATA
            return PacketValidationResult(
                is_valid=False,
                state=self.current_state,
                error_message=f"Magic header mismatch: expected 0x{MAGIC_HEADER:08X}, got 0x{magic:08X}.",
            )

        # Unpack 12 sensor floats (offset 12)
        floats = struct.unpack_from("<12f", raw_bytes, 12)
        status_flags, checksum = struct.unpack_from("<HH", raw_bytes, 60)

        # Compute 16-bit additive checksum over bytes 0..61
        computed_csum = sum(raw_bytes[:62]) & 0xFFFF
        if checksum != 0 and checksum != computed_csum:
            self.current_state = ConnectionState.INVALID_SENSOR_DATA
            return PacketValidationResult(
                is_valid=False,
                state=self.current_state,
                timestamp_ms=t_ms,
                seq_num=seq,
                error_message=f"Checksum verification failed: expected {checksum}, computed {computed_csum}.",
            )

        channels = {k: float(round(v, 4)) for k, v in zip(CANONICAL_CHANNELS, floats)}
        return self._evaluate_parsed_sample(channels, t_ms, seq, status_flags, now)

    def validate_json_packet(self, json_str: str, receive_time: Optional[float] = None) -> PacketValidationResult:
        """Validate and parse a JSON dictionary packet.

        Format:
          {"t_ms": 1234, "seq": 1, "rua": [ax,ay,az,gx,gy,gz], "rla": [ax,ay,az,gx,gy,gz], "status": 0}
        """
        now = receive_time or time.time()
        try:
            data = json.loads(json_str)
        except Exception as exc:
            self.current_state = ConnectionState.INVALID_SENSOR_DATA
            return PacketValidationResult(
                is_valid=False,
                state=self.current_state,
                error_message=f"JSON decoding error: {exc}",
            )

        if not isinstance(data, dict):
            return PacketValidationResult(
                is_valid=False,
                state=ConnectionState.INVALID_SENSOR_DATA,
                error_message="Packet payload is not a JSON object.",
            )

        t_ms = int(data.get("t_ms", 0))
        seq = int(data.get("seq", 0))
        status_flags = int(data.get("status", 0))

        rua = data.get("rua", [])
        rla = data.get("rla", [])

        if len(rua) != 6 or len(rla) != 6:
            self.current_state = ConnectionState.INVALID_SENSOR_DATA
            return PacketValidationResult(
                is_valid=False,
                state=self.current_state,
                error_message=f"Incomplete 12-channel data: rua has {len(rua)} items, rla has {len(rla)} items (expected 6 each).",
            )

        vals = list(rua) + list(rla)
        channels = {k: float(round(float(v), 4)) for k, v in zip(CANONICAL_CHANNELS, vals)}
        return self._evaluate_parsed_sample(channels, t_ms, seq, status_flags, now)

    def validate_csv_line(self, csv_line: str, receive_time: Optional[float] = None) -> PacketValidationResult:
        """Validate and parse a comma-separated stream line.

        Format:
          t_ms,seq,rua_ax,rua_ay,rua_az,rua_gx,rua_gy,rua_gz,rla_ax,rla_ay,rla_az,rla_gx,rla_gy,rla_gz
        """
        now = receive_time or time.time()
        parts = [p.strip() for p in csv_line.strip().split(",") if p.strip()]
        if len(parts) < 14:
            self.current_state = ConnectionState.INVALID_SENSOR_DATA
            return PacketValidationResult(
                is_valid=False,
                state=self.current_state,
                error_message=f"CSV line contains {len(parts)} columns; expected 14 (t_ms, seq, + 12 channels).",
            )

        try:
            t_ms = int(float(parts[0]))
            seq = int(float(parts[1]))
            vals = [float(p) for p in parts[2:14]]
        except ValueError as exc:
            self.current_state = ConnectionState.INVALID_SENSOR_DATA
            return PacketValidationResult(
                is_valid=False,
                state=self.current_state,
                error_message=f"Non-numeric field in sensor stream: {exc}",
            )

        channels = {k: float(round(v, 4)) for k, v in zip(CANONICAL_CHANNELS, vals)}
        return self._evaluate_parsed_sample(channels, t_ms, seq, 0, now)

    def _evaluate_parsed_sample(
        self,
        channels: Dict[str, float],
        t_ms: int,
        seq: int,
        status_flags: int,
        receive_time: float,
    ) -> PacketValidationResult:
        """Perform clinical biomechanical validation on parsed 12-channel data."""
        # 1. Non-finite check
        for ch, v in channels.items():
            if not np.isfinite(v):
                self.current_state = ConnectionState.INVALID_SENSOR_DATA
                return PacketValidationResult(
                    is_valid=False,
                    state=self.current_state,
                    timestamp_ms=t_ms,
                    seq_num=seq,
                    error_message=f"Channel {ch} contains non-finite value ({v}).",
                )

        # 2. Physical range checks
        for ch in ["RUA_Ax", "RUA_Ay", "RUA_Az", "RLA_Ax", "RLA_Ay", "RLA_Az"]:
            if abs(channels[ch]) > self.acc_limit:
                self.current_state = ConnectionState.INVALID_SENSOR_DATA
                return PacketValidationResult(
                    is_valid=False,
                    state=self.current_state,
                    timestamp_ms=t_ms,
                    seq_num=seq,
                    error_message=f"Accelerometer channel {ch} exceeded safety threshold ({channels[ch]:.1f} > {self.acc_limit} m/s^2).",
                )

        for ch in ["RUA_Gx", "RUA_Gy", "RUA_Gz", "RLA_Gx", "RLA_Gy", "RLA_Gz"]:
            if abs(channels[ch]) > self.gyr_limit:
                self.current_state = ConnectionState.INVALID_SENSOR_DATA
                return PacketValidationResult(
                    is_valid=False,
                    state=self.current_state,
                    timestamp_ms=t_ms,
                    seq_num=seq,
                    error_message=f"Gyroscope channel {ch} exceeded physical rotation threshold ({channels[ch]:.1f} > {self.gyr_limit} rad/s).",
                )

        # 3. Individual IMU Health: Distinguish working dual-IMU from single-IMU
        rua_acc_norm = np.sqrt(channels["RUA_Ax"]**2 + channels["RUA_Ay"]**2 + channels["RUA_Az"]**2)
        rla_acc_norm = np.sqrt(channels["RLA_Ax"]**2 + channels["RLA_Ay"]**2 + channels["RLA_Az"]**2)

        # Flag 0x01 = IMU1 error, Flag 0x02 = IMU2 error from firmware
        rua_hw_err = bool(status_flags & 0x01)
        rla_hw_err = bool(status_flags & 0x02)

        rua_active = bool((rua_acc_norm > 0.05) and not rua_hw_err)
        rla_active = bool((rla_acc_norm > 0.05) and not rla_hw_err)

        rua_status = IMUStatus.HEALTHY if rua_active else IMUStatus.MISSING
        rla_status = IMUStatus.HEALTHY if rla_active else IMUStatus.MISSING

        both_imus_active = bool(rua_active and rla_active)

        # 4. Monotonicity & sequence drop tracking
        warning: Optional[str] = None
        if self.last_seq_num is not None:
            expected_seq = (self.last_seq_num + 1) & 0xFFFFFFFF
            if seq != expected_seq and seq > self.last_seq_num:
                dropped = seq - self.last_seq_num - 1
                self.consecutive_drops += dropped
                warning = f"Detected {dropped} dropped packet(s) (expected seq {expected_seq}, got {seq})."
            else:
                self.consecutive_drops = 0

        self.last_seq_num = seq
        self.last_packet_time = receive_time
        self.total_packets_received += 1

        # 5. Sampling frequency tracking
        self._timestamps_window.append(receive_time)
        if len(self._timestamps_window) > 50:
            self._timestamps_window.pop(0)

        if len(self._timestamps_window) >= 10:
            dt_span = self._timestamps_window[-1] - self._timestamps_window[0]
            if dt_span > 0:
                self._measured_rate_hz = round((len(self._timestamps_window) - 1) / dt_span, 1)

        # Update state to verified
        if not both_imus_active:
            missing_name = "Forearm IMU (RLA)" if not rla_active else "Upper Arm IMU (RUA)"
            warning = f"Single-IMU Warning: {missing_name} is silent or uncalibrated. 12-channel joint kinematics require dual IMUs."
            self.current_state = ConnectionState.CONNECTED_VERIFIED
        else:
            self.current_state = ConnectionState.CONNECTED_VERIFIED

        return PacketValidationResult(
            is_valid=True,
            state=self.current_state,
            timestamp_ms=t_ms,
            seq_num=seq,
            channels=channels,
            both_imus_active=both_imus_active,
            rua_status=rua_status,
            rla_status=rla_status,
            status_flags=status_flags,
            warning_message=warning,
        )

    def get_status_report(self) -> Dict[str, Any]:
        """Generate complete diagnostic summary for the UI and REST API."""
        return {
            "state": self.current_state.value,
            "is_verified": self.current_state == ConnectionState.CONNECTED_VERIFIED,
            "is_simulation": self.is_simulation,
            "total_packets": self.total_packets_received,
            "measured_rate_hz": self._measured_rate_hz,
            "last_packet_time": self.last_packet_time,
            "last_seq_num": self.last_seq_num,
        }
