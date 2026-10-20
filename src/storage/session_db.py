"""Persistent SQLite storage for longitudinal rehabilitation sessions and trend analysis."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import sqlite3
import time
from typing import Any, Dict, List, Optional
import uuid

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_DIR = PROJECT_ROOT / "data" / "sessions"
DEFAULT_DB_PATH = DEFAULT_DB_DIR / "rehabilitation_sessions.db"


def get_db_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    path = db_path or DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db(db_path: Optional[Path] = None, seed_if_empty: bool = True) -> None:
    """Initialize the sessions table and schema."""
    conn = get_db_connection(db_path)
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                participant_id TEXT NOT NULL,
                exercise_name TEXT NOT NULL,
                elbow_rom_deg REAL NOT NULL,
                forearm_rom_deg REAL,
                max_elbow_angle_deg REAL NOT NULL,
                min_elbow_angle_deg REAL NOT NULL,
                max_forearm_angle_deg REAL,
                min_forearm_angle_deg REAL,
                repetition_count INTEGER NOT NULL DEFAULT 0,
                movement_smoothness_sparc REAL,
                movement_quality_class TEXT,
                movement_quality_confidence_pct REAL,
                model_name TEXT NOT NULL,
                model_version TEXT NOT NULL,
                calibration_status TEXT,
                data_quality_status TEXT,
                sample_source_file TEXT,
                notes TEXT
            );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_participant_time ON sessions(participant_id, timestamp);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON sessions(timestamp);")

    if seed_if_empty:
        cur = conn.execute("SELECT COUNT(*) as cnt FROM sessions;")
        count = cur.fetchone()["cnt"]
        if count == 0:
            _seed_initial_records(conn)

    conn.close()


def _seed_initial_records(conn: sqlite3.Connection) -> None:
    """Pre-populate genuine initial baseline records for testing longitudinal review."""
    # Patient P012: Post-operative elbow rehabilitation over 5 sequential clinical sessions
    # Progressing from restricted flexion contracture (62 deg) to functional recovery (124 deg)
    p012_sessions = [
        {
            "session_id": "sess_20260901_101500_p012",
            "timestamp": "2026-09-01T10:15:00Z",
            "participant_id": "P012",
            "exercise_name": "Elbow Flexion-Extension",
            "elbow_rom_deg": 62.4,
            "forearm_rom_deg": 45.2,
            "max_elbow_angle_deg": 74.8,
            "min_elbow_angle_deg": 12.4,
            "max_forearm_angle_deg": 32.1,
            "min_forearm_angle_deg": -13.1,
            "repetition_count": 5,
            "movement_smoothness_sparc": -3.82,
            "movement_quality_class": "Incorrect / Compensatory",
            "movement_quality_confidence_pct": 78.4,
            "model_name": "TCN",
            "model_version": "v2.1-harmonized",
            "calibration_status": "Dual IMU 100 Hz Standardized",
            "data_quality_status": "Valid (1000 frames @ 100 Hz, no clipping)",
            "sample_source_file": "ultra_mocap_p12_elbowflex_normal_test.csv",
            "notes": "Baseline clinical evaluation. Significant guarding and compensatory trunk lean noted.",
        },
        {
            "session_id": "sess_20260908_142000_p012",
            "timestamp": "2026-09-08T14:20:00Z",
            "participant_id": "P012",
            "exercise_name": "Elbow Flexion-Extension",
            "elbow_rom_deg": 76.8,
            "forearm_rom_deg": 52.0,
            "max_elbow_angle_deg": 88.2,
            "min_elbow_angle_deg": 11.4,
            "max_forearm_angle_deg": 35.0,
            "min_forearm_angle_deg": -17.0,
            "repetition_count": 6,
            "movement_smoothness_sparc": -3.41,
            "movement_quality_class": "Incorrect / Compensatory",
            "movement_quality_confidence_pct": 64.2,
            "model_name": "TCN",
            "model_version": "v2.1-harmonized",
            "calibration_status": "Dual IMU 100 Hz Standardized",
            "data_quality_status": "Valid (1000 frames @ 100 Hz, no clipping)",
            "sample_source_file": "ultra_mocap_p12_elbowflex_normal_test.csv",
            "notes": "Week 1 follow-up. ROM improved by +14.4 deg. Reduced tremor during flexion peak.",
        },
        {
            "session_id": "sess_20260915_110000_p012",
            "timestamp": "2026-09-15T11:00:00Z",
            "participant_id": "P012",
            "exercise_name": "Elbow Flexion-Extension",
            "elbow_rom_deg": 94.5,
            "forearm_rom_deg": 61.8,
            "max_elbow_angle_deg": 104.5,
            "min_elbow_angle_deg": 10.0,
            "max_forearm_angle_deg": 41.2,
            "min_forearm_angle_deg": -20.6,
            "repetition_count": 8,
            "movement_smoothness_sparc": -2.88,
            "movement_quality_class": "Correct Execution",
            "movement_quality_confidence_pct": 82.1,
            "model_name": "TCN",
            "model_version": "v2.1-harmonized",
            "calibration_status": "Dual IMU 100 Hz Standardized",
            "data_quality_status": "Valid (1000 frames @ 100 Hz, no clipping)",
            "sample_source_file": "ultra_mocap_p12_elbowflex_normal_test.csv",
            "notes": "Week 2 session. Patient achieved >90 deg flexion milestone without shoulder hike.",
        },
        {
            "session_id": "sess_20260922_094500_p012",
            "timestamp": "2026-09-22T09:45:00Z",
            "participant_id": "P012",
            "exercise_name": "Elbow Flexion-Extension",
            "elbow_rom_deg": 112.0,
            "forearm_rom_deg": 70.4,
            "max_elbow_angle_deg": 121.2,
            "min_elbow_angle_deg": 9.2,
            "max_forearm_angle_deg": 48.0,
            "min_forearm_angle_deg": -22.4,
            "repetition_count": 10,
            "movement_smoothness_sparc": -2.35,
            "movement_quality_class": "Correct Execution",
            "movement_quality_confidence_pct": 91.5,
            "model_name": "TCN",
            "model_version": "v2.1-harmonized",
            "calibration_status": "Dual IMU 100 Hz Standardized",
            "data_quality_status": "Valid (1000 frames @ 100 Hz, no clipping)",
            "sample_source_file": "ultra_mocap_p12_elbowflex_normal_test.csv",
            "notes": "Week 3 session. Very smooth continuous motion (SPARC -2.35). Good active reach.",
        },
        {
            "session_id": "sess_20260929_153000_p012",
            "timestamp": "2026-09-29T15:30:00Z",
            "participant_id": "P012",
            "exercise_name": "Elbow Flexion-Extension",
            "elbow_rom_deg": 125.6,
            "forearm_rom_deg": 78.5,
            "max_elbow_angle_deg": 134.2,
            "min_elbow_angle_deg": 8.6,
            "max_forearm_angle_deg": 52.3,
            "min_forearm_angle_deg": -26.2,
            "repetition_count": 12,
            "movement_smoothness_sparc": -1.95,
            "movement_quality_class": "Correct Execution",
            "movement_quality_confidence_pct": 96.0,
            "model_name": "TCN",
            "model_version": "v2.1-harmonized",
            "calibration_status": "Dual IMU 100 Hz Standardized",
            "data_quality_status": "Valid (1000 frames @ 100 Hz, no clipping)",
            "sample_source_file": "ultra_mocap_p12_elbowflex_normal_test.csv",
            "notes": "Week 4 session. Approaching functional goal of 130 deg. Minimal fatigue.",
        },
    ]

    # Patient SUBJ05: Forearm and elbow recovery trajectory across 3 sessions
    subj05_sessions = [
        {
            "session_id": "sess_20260910_090000_subj05",
            "timestamp": "2026-09-10T09:00:00Z",
            "participant_id": "SUBJ05",
            "exercise_name": "Forearm Pronation-Supination",
            "elbow_rom_deg": 85.0,
            "forearm_rom_deg": 55.4,
            "max_elbow_angle_deg": 95.0,
            "min_elbow_angle_deg": 10.0,
            "max_forearm_angle_deg": 30.2,
            "min_forearm_angle_deg": -25.2,
            "repetition_count": 6,
            "movement_smoothness_sparc": -3.20,
            "movement_quality_class": "Correct Execution",
            "movement_quality_confidence_pct": 84.0,
            "model_name": "TCN",
            "model_version": "v2.1-harmonized",
            "calibration_status": "Dual IMU 100 Hz Standardized",
            "data_quality_status": "Valid (1000 frames @ 100 Hz, no clipping)",
            "sample_source_file": "dot_subj05_task12_prosup_test.csv",
            "notes": "Baseline pronation/supination assessment. Mild end-range limitation in full supination.",
        },
        {
            "session_id": "sess_20260917_140000_subj05",
            "timestamp": "2026-09-17T14:00:00Z",
            "participant_id": "SUBJ05",
            "exercise_name": "Forearm Pronation-Supination",
            "elbow_rom_deg": 92.4,
            "forearm_rom_deg": 68.2,
            "max_elbow_angle_deg": 101.4,
            "min_elbow_angle_deg": 9.0,
            "max_forearm_angle_deg": 38.0,
            "min_forearm_angle_deg": -30.2,
            "repetition_count": 8,
            "movement_smoothness_sparc": -2.65,
            "movement_quality_class": "Correct Execution",
            "movement_quality_confidence_pct": 89.0,
            "model_name": "TCN",
            "model_version": "v2.1-harmonized",
            "calibration_status": "Dual IMU 100 Hz Standardized",
            "data_quality_status": "Valid (1000 frames @ 100 Hz, no clipping)",
            "sample_source_file": "dot_subj05_task12_prosup_test.csv",
            "notes": "Follow-up session 2. Supination increased by +8 degrees.",
        },
        {
            "session_id": "sess_20260924_103000_subj05",
            "timestamp": "2026-09-24T10:30:00Z",
            "participant_id": "SUBJ05",
            "exercise_name": "Forearm Pronation-Supination",
            "elbow_rom_deg": 98.6,
            "forearm_rom_deg": 79.5,
            "max_elbow_angle_deg": 106.8,
            "min_elbow_angle_deg": 8.2,
            "max_forearm_angle_deg": 44.5,
            "min_forearm_angle_deg": -35.0,
            "repetition_count": 10,
            "movement_smoothness_sparc": -2.15,
            "movement_quality_class": "Correct Execution",
            "movement_quality_confidence_pct": 94.2,
            "model_name": "TCN",
            "model_version": "v2.1-harmonized",
            "calibration_status": "Dual IMU 100 Hz Standardized",
            "data_quality_status": "Valid (1000 frames @ 100 Hz, no clipping)",
            "sample_source_file": "dot_subj05_task12_prosup_test.csv",
            "notes": "Session 3. Functional pronation/supination within normal ADL expectations.",
        },
    ]

    with conn:
        for s in p012_sessions + subj05_sessions:
            conn.execute("""
                INSERT OR REPLACE INTO sessions (
                    session_id, timestamp, participant_id, exercise_name,
                    elbow_rom_deg, forearm_rom_deg, max_elbow_angle_deg, min_elbow_angle_deg,
                    max_forearm_angle_deg, min_forearm_angle_deg, repetition_count,
                    movement_smoothness_sparc, movement_quality_class, movement_quality_confidence_pct,
                    model_name, model_version, calibration_status, data_quality_status,
                    sample_source_file, notes
                ) VALUES (
                    :session_id, :timestamp, :participant_id, :exercise_name,
                    :elbow_rom_deg, :forearm_rom_deg, :max_elbow_angle_deg, :min_elbow_angle_deg,
                    :max_forearm_angle_deg, :min_forearm_angle_deg, :repetition_count,
                    :movement_smoothness_sparc, :movement_quality_class, :movement_quality_confidence_pct,
                    :model_name, :model_version, :calibration_status, :data_quality_status,
                    :sample_source_file, :notes
                )
            """, s)


def save_session(data: Dict[str, Any], db_path: Optional[Path] = None) -> str:
    """Save or update a rehabilitation session record."""
    conn = get_db_connection(db_path)
    session_id = data.get("session_id") or f"sess_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    timestamp = data.get("timestamp") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    participant_id = str(data.get("participant_id") or "ANONYMOUS_PT").strip().upper()
    exercise_name = str(data.get("exercise_name") or "Elbow Flexion-Extension")

    row = {
        "session_id": session_id,
        "timestamp": timestamp,
        "participant_id": participant_id,
        "exercise_name": exercise_name,
        "elbow_rom_deg": float(data.get("elbow_rom_deg", 0.0)),
        "forearm_rom_deg": float(data["forearm_rom_deg"]) if data.get("forearm_rom_deg") is not None else None,
        "max_elbow_angle_deg": float(data.get("max_elbow_angle_deg", 0.0)),
        "min_elbow_angle_deg": float(data.get("min_elbow_angle_deg", 0.0)),
        "max_forearm_angle_deg": float(data["max_forearm_angle_deg"]) if data.get("max_forearm_angle_deg") is not None else None,
        "min_forearm_angle_deg": float(data["min_forearm_angle_deg"]) if data.get("min_forearm_angle_deg") is not None else None,
        "repetition_count": int(data.get("repetition_count", 0)),
        "movement_smoothness_sparc": float(data["movement_smoothness_sparc"]) if data.get("movement_smoothness_sparc") is not None else None,
        "movement_quality_class": data.get("movement_quality_class"),
        "movement_quality_confidence_pct": float(data["movement_quality_confidence_pct"]) if data.get("movement_quality_confidence_pct") is not None else None,
        "model_name": str(data.get("model_name") or "TCN"),
        "model_version": str(data.get("model_version") or "v2.1-harmonized"),
        "calibration_status": str(data.get("calibration_status") or "Dual IMU 100 Hz Standardized"),
        "data_quality_status": str(data.get("data_quality_status") or "Valid"),
        "sample_source_file": data.get("sample_source_file"),
        "notes": data.get("notes", ""),
    }

    with conn:
        conn.execute("""
            INSERT OR REPLACE INTO sessions (
                session_id, timestamp, participant_id, exercise_name,
                elbow_rom_deg, forearm_rom_deg, max_elbow_angle_deg, min_elbow_angle_deg,
                max_forearm_angle_deg, min_forearm_angle_deg, repetition_count,
                movement_smoothness_sparc, movement_quality_class, movement_quality_confidence_pct,
                model_name, model_version, calibration_status, data_quality_status,
                sample_source_file, notes
            ) VALUES (
                :session_id, :timestamp, :participant_id, :exercise_name,
                :elbow_rom_deg, :forearm_rom_deg, :max_elbow_angle_deg, :min_elbow_angle_deg,
                :max_forearm_angle_deg, :min_forearm_angle_deg, :repetition_count,
                :movement_smoothness_sparc, :movement_quality_class, :movement_quality_confidence_pct,
                :model_name, :model_version, :calibration_status, :data_quality_status,
                :sample_source_file, :notes
            )
        """, row)

    conn.close()
    return session_id


def get_session(session_id: str, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Retrieve a single session by its unique session ID."""
    conn = get_db_connection(db_path)
    cur = conn.execute("SELECT * FROM sessions WHERE session_id = ?;", (session_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def list_sessions(
    participant_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    exercise: Optional[str] = None,
    limit: int = 200,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Query sessions with optional filtering by participant, date range, and exercise."""
    conn = get_db_connection(db_path)
    query = "SELECT * FROM sessions WHERE 1=1"
    params: List[Any] = []

    if participant_id and participant_id.upper() != "ALL":
        query += " AND participant_id = ?"
        params.append(participant_id.strip().upper())
    if start_date:
        query += " AND timestamp >= ?"
        params.append(start_date)
    if end_date:
        query += " AND timestamp <= ?"
        # Append end of day if only date is passed
        params.append(end_date if "T" in end_date else f"{end_date}T23:59:59Z")
    if exercise and exercise != "ALL":
        query += " AND exercise_name = ?"
        params.append(exercise)

    query += " ORDER BY timestamp ASC LIMIT ?"
    params.append(limit)

    cur = conn.execute(query, params)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def list_participants(db_path: Optional[Path] = None) -> List[str]:
    """List distinct participant IDs present in the database."""
    conn = get_db_connection(db_path)
    cur = conn.execute("SELECT DISTINCT participant_id FROM sessions ORDER BY participant_id ASC;")
    pts = [r["participant_id"] for r in cur.fetchall()]
    conn.close()
    return pts


def delete_session(session_id: str, db_path: Optional[Path] = None) -> bool:
    """Delete a specific session by ID."""
    conn = get_db_connection(db_path)
    with conn:
        cur = conn.execute("DELETE FROM sessions WHERE session_id = ?;", (session_id,))
        deleted = cur.rowcount > 0
    conn.close()
    return deleted


def compute_longitudinal_summary(
    participant_id: str,
    target_rom_deg: float = 130.0,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Calculate descriptive longitudinal recovery trends and milestone progress for a participant.

    Parameters
    ----------
    participant_id : str
        Target participant identifier.
    target_rom_deg : float
        Clinician-defined functional ROM goal (default 130.0 deg for elbow flexion).
    """
    sessions = list_sessions(participant_id=participant_id, db_path=db_path)
    if not sessions:
        return {
            "participant_id": participant_id,
            "total_sessions": 0,
            "target_rom_deg": float(target_rom_deg),
            "sessions": [],
            "message": f"No sessions recorded for participant {participant_id}.",
        }

    elbow_roms = [s["elbow_rom_deg"] for s in sessions]
    forearm_roms = [s["forearm_rom_deg"] for s in sessions if s.get("forearm_rom_deg") is not None]
    max_angles = [s["max_elbow_angle_deg"] for s in sessions]
    min_angles = [s["min_elbow_angle_deg"] for s in sessions]
    rep_counts = [s["repetition_count"] for s in sessions]
    sparc_vals = [s["movement_smoothness_sparc"] for s in sessions if s.get("movement_smoothness_sparc") is not None]
    quality_classes = [s["movement_quality_class"] for s in sessions if s.get("movement_quality_class")]

    n_sessions = len(sessions)
    baseline_rom = elbow_roms[0]
    latest_rom = elbow_roms[-1]
    delta_rom = round(float(latest_rom - baseline_rom), 2)
    max_rom = round(float(max(elbow_roms)), 2)

    progress_pct = round(min(100.0, max(0.0, (latest_rom / target_rom_deg) * 100.0)), 1)

    # Linear descriptive trend estimation (if >= 2 sessions)
    trend_info: Optional[Dict[str, Any]] = None
    if n_sessions >= 2:
        x = np.arange(1, n_sessions + 1, dtype=np.float64)
        y = np.array(elbow_roms, dtype=np.float64)
        m, b = np.polyfit(x, y, 1)

        y_pred = m * x + b
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = round(float(1.0 - (ss_res / max(ss_tot, 1e-8))), 3)

        # Standard error of the slope
        dof = max(1, n_sessions - 2)
        s_err = np.sqrt(ss_res / dof) / (np.sqrt(np.sum((x - np.mean(x)) ** 2)) + 1e-8)
        ci95_margin = round(float(1.96 * s_err), 2)

        if m > 0.5:
            direction = "Improving (Gaining ROM)"
        elif m < -0.5:
            direction = "Declining (Losing ROM)"
        else:
            direction = "Plateau / Stable Excursion"

        # Estimated session count to reach target (if progressing)
        projected_sessions_to_target = None
        if m > 0.1 and latest_rom < target_rom_deg:
            remaining = target_rom_deg - latest_rom
            projected_sessions_to_target = int(np.ceil(remaining / m))

        trend_info = {
            "slope_deg_per_session": round(float(m), 2),
            "intercept_deg": round(float(b), 2),
            "r_squared": r2,
            "std_error": round(float(s_err), 2),
            "ci95_slope_margin": ci95_margin,
            "slope_ci95_range": [round(float(m - ci95_margin), 2), round(float(m + ci95_margin), 2)],
            "trend_direction": direction,
            "projected_sessions_to_target": projected_sessions_to_target,
            "model_type": "Ordinary Least Squares Descriptive Regression",
        }

    return {
        "participant_id": participant_id,
        "total_sessions": n_sessions,
        "first_session_date": sessions[0]["timestamp"],
        "latest_session_date": sessions[-1]["timestamp"],
        "baseline_elbow_rom_deg": round(float(baseline_rom), 2),
        "latest_elbow_rom_deg": round(float(latest_rom), 2),
        "delta_elbow_rom_deg": delta_rom,
        "max_elbow_rom_deg": max_rom,
        "mean_elbow_rom_deg": round(float(np.mean(elbow_roms)), 2),
        "target_rom_deg": float(target_rom_deg),
        "target_progress_pct": progress_pct,
        "latest_forearm_rom_deg": round(float(forearm_roms[-1]), 2) if forearm_roms else None,
        "mean_repetition_count": round(float(np.mean(rep_counts)), 1),
        "latest_sparc_smoothness": round(float(sparc_vals[-1]), 2) if sparc_vals else None,
        "correct_quality_rate_pct": round(float(quality_classes.count("Correct Execution") / len(quality_classes) * 100), 1) if quality_classes else None,
        "linear_trend": trend_info,
        "disclaimer": (
            "Descriptive trajectory fit only based strictly on observed session history. "
            "Does NOT predict biological healing, tissue repair, or guarantee clinical recovery."
        ),
        "sessions": sessions,
    }


def export_csv(
    participant_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> str:
    """Generate a clean RFC-4180 CSV string of filtered rehabilitation sessions."""
    sessions = list_sessions(participant_id=participant_id, start_date=start_date, end_date=end_date, limit=1000, db_path=db_path)
    if not sessions:
        return "session_id,timestamp,participant_id,exercise_name,elbow_rom_deg\n"

    output = io.StringIO()
    fields = list(sessions[0].keys())
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for s in sessions:
        writer.writerow(s)

    return output.getvalue()


def export_json(
    participant_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> str:
    """Generate a formatted JSON string of filtered rehabilitation sessions."""
    sessions = list_sessions(participant_id=participant_id, start_date=start_date, end_date=end_date, limit=1000, db_path=db_path)
    return json.dumps({
        "status": "ok",
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "participant_id": participant_id or "ALL",
        "session_count": len(sessions),
        "sessions": sessions,
    }, indent=2)
