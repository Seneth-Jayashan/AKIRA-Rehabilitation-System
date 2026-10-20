"""FastAPI backend application for the AI Elbow Rehabilitation Demo."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from demo.backend.engine import RehabilitationEngine
from src.storage.session_db import (
    init_db,
    save_session,
    get_session,
    list_sessions,
    list_participants,
    delete_session,
    compute_longitudinal_summary,
    export_csv,
    export_json,
)
from src.inference.esp32_live_validator import (
    ESP32LiveStreamValidator,
    ConnectionState,
    IMUStatus,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = PROJECT_ROOT / "demo" / "frontend"

engine = RehabilitationEngine(project_root=PROJECT_ROOT)
esp32_validator = ESP32LiveStreamValidator()
init_db()

app = FastAPI(
    title="AI Elbow Rehabilitation Assistant — Interactive Model Testing Demo",
    description=(
        "Comparative evaluation of TCN, 1D-CNN, LSTM, and GRU on standardized dual-IMU rehabilitation data. "
        "Supports Continuous Elbow Flexion Angle & ROM, Forearm Pronation/Supination & ROM, "
        "Exercise Classification, and Movement Quality Assessment."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static frontend assets
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
    app.mount("/static_v2", StaticFiles(directory=str(FRONTEND_DIR)), name="static_v2")


class SampleAnalysisRequest(BaseModel):
    sample_id: str
    selected_model: str = "COMPARE_ALL"


@app.get("/", response_class=HTMLResponse)
def serve_dashboard() -> HTMLResponse:
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Demo index.html not found.")
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"))


@app.get("/api/health")
@app.get("/api/v2/health")
def health_check() -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": "AI Elbow Rehabilitation Inference Engine",
        "supported_models": ["TCN", "1D_CNN", "LSTM", "GRU", "COMPARE_ALL"],
        "supported_tasks": [
            "Continuous Elbow Flexion Angle Regression & ROM (deg)",
            "Continuous Forearm Pronation/Supination Angle Regression & ROM (deg)",
            "Upper-Limb Rehabilitation Exercise Classification (7-Class Harmonized Taxonomy)",
            "Exercise Movement-Quality Classification (Binary: Correct vs Incorrect)",
            "Three-Tier Range of Motion & Repetition Cycle Detection",
        ],
        "input_channels": 12,
        "sampling_rate_hz": 100.0,
    }


@app.get("/api/overview")
@app.get("/api/v2/overview")
def get_overview() -> Dict[str, Any]:
    manifest = engine.get_split_manifest()
    summary = engine.get_benchmark_summary()
    return {
        "manifest": manifest,
        "summary": summary,
        "samples_count": len(engine.list_samples()),
    }


@app.get("/api/samples")
@app.get("/api/v2/samples")
def list_samples() -> Dict[str, Any]:
    return {"samples": engine.list_samples()}


@app.get("/api/samples/{sample_id}/download")
@app.get("/api/v2/samples/{sample_id}/download")
def download_sample_csv(sample_id: str) -> FileResponse:
    sample_meta = next((s for s in engine.list_samples() if s["sample_id"] == sample_id), None)
    if sample_meta is None:
        raise HTTPException(status_code=404, detail=f"Sample {sample_id!r} not found.")
    csv_path = engine.samples_dir / sample_meta["filename"]
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="Sample CSV not found on disk.")
    return FileResponse(path=str(csv_path), media_type="text/csv", filename=sample_meta["filename"])


@app.post("/api/analyze/sample")
@app.post("/api/v2/analyze/sample")
def analyze_sample(req: SampleAnalysisRequest) -> Dict[str, Any]:
    sample_meta = next((s for s in engine.list_samples() if s["sample_id"] == req.sample_id), None)
    if sample_meta is None:
        raise HTTPException(status_code=404, detail=f"Sample {req.sample_id!r} not found.")

    csv_path = engine.samples_dir / sample_meta["filename"]
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail=f"File {sample_meta['filename']!r} not found on disk.")

    with open(csv_path, "rb") as f:
        csv_bytes = f.read()

    try:
        results = engine.run_complete_analysis(
            csv_bytes=csv_bytes,
            filename=sample_meta["filename"],
            selected_model=req.selected_model,
        )
        results["sample_metadata"] = sample_meta
        return results
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc


@app.post("/api/analyze/upload")
@app.post("/api/v2/analyze/upload")
async def analyze_upload(
    file: UploadFile = File(...),
    selected_model: str = Form("COMPARE_ALL"),
) -> Dict[str, Any]:
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a .csv file.")

    csv_bytes = await file.read()
    try:
        results = engine.run_complete_analysis(
            csv_bytes=csv_bytes,
            filename=file.filename,
            selected_model=selected_model,
        )
        results["sample_metadata"] = {
            "sample_id": f"uploaded_{file.filename}",
            "filename": file.filename,
            "source_dataset": "User Upload",
            "subject_id": "Unknown",
            "exercise_name": "Custom Uploaded Trial",
            "has_ground_truth": results["ground_truth_metadata"]["has_ground_truth"],
        }
        return results
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to process uploaded CSV: {exc}") from exc


# =============================================================================
# Longitudinal Rehabilitation Tracking Endpoints
# =============================================================================

class SaveSessionRequest(BaseModel):
    participant_id: str
    exercise_name: Optional[str] = "Elbow Flexion-Extension"
    elbow_rom_deg: float
    forearm_rom_deg: Optional[float] = None
    max_elbow_angle_deg: float
    min_elbow_angle_deg: float
    max_forearm_angle_deg: Optional[float] = None
    min_forearm_angle_deg: Optional[float] = None
    repetition_count: Optional[int] = 0
    movement_smoothness_sparc: Optional[float] = None
    movement_quality_class: Optional[str] = None
    movement_quality_confidence_pct: Optional[float] = None
    model_name: Optional[str] = "TCN"
    model_version: Optional[str] = "v2.1-harmonized"
    calibration_status: Optional[str] = "Dual IMU 100 Hz Standardized"
    data_quality_status: Optional[str] = "Valid"
    sample_source_file: Optional[str] = None
    notes: Optional[str] = None


@app.get("/api/longitudinal/participants")
def api_list_participants() -> Dict[str, Any]:
    """Retrieve distinct participant IDs recorded in session history."""
    pts = list_participants()
    return {"status": "ok", "participants": pts}


@app.get("/api/longitudinal/sessions")
def api_list_sessions(
    participant_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    exercise: Optional[str] = None,
) -> Dict[str, Any]:
    """List recorded rehabilitation sessions with optional filtering."""
    sessions = list_sessions(
        participant_id=participant_id,
        start_date=start_date,
        end_date=end_date,
        exercise=exercise,
    )
    return {
        "status": "ok",
        "count": len(sessions),
        "participant_id": participant_id or "ALL",
        "sessions": sessions,
    }


@app.post("/api/longitudinal/sessions")
def api_save_session(req: SaveSessionRequest) -> Dict[str, Any]:
    """Save an analyzed rehabilitation session to persistent SQLite storage."""
    try:
        data = req.model_dump() if hasattr(req, "model_dump") else req.dict()
        session_id = save_session(data)
        return {
            "status": "ok",
            "message": "Session saved successfully.",
            "session_id": session_id,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to save session: {exc}") from exc


@app.get("/api/longitudinal/sessions/{session_id}")
def api_get_session(session_id: str) -> Dict[str, Any]:
    """Retrieve a single session by its unique ID."""
    s = get_session(session_id)
    if not s:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found.")
    return {"status": "ok", "session": s}


@app.delete("/api/longitudinal/sessions/{session_id}")
def api_delete_session(session_id: str) -> Dict[str, Any]:
    """Delete a specific session by its unique ID."""
    deleted = delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found.")
    return {"status": "ok", "message": f"Session {session_id} deleted."}


@app.get("/api/longitudinal/summary/{participant_id}")
def api_participant_summary(
    participant_id: str,
    target_rom_deg: float = 130.0,
) -> Dict[str, Any]:
    """Calculate descriptive longitudinal recovery trends and target progress."""
    summary = compute_longitudinal_summary(
        participant_id=participant_id,
        target_rom_deg=target_rom_deg,
    )
    return {"status": "ok", "summary": summary}


@app.get("/api/longitudinal/export/csv")
def api_export_csv(
    participant_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Response:
    """Download sessions history formatted as CSV."""
    csv_text = export_csv(participant_id=participant_id, start_date=start_date, end_date=end_date)
    return Response(
        content=csv_text,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=rehabilitation_sessions_{participant_id or 'all'}.csv"},
    )


@app.get("/api/longitudinal/export/json")
def api_export_json(
    participant_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Response:
    """Download sessions history formatted as JSON."""
    json_text = export_json(participant_id=participant_id, start_date=start_date, end_date=end_date)
    return Response(
        content=json_text,
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename=rehabilitation_sessions_{participant_id or 'all'}.json"},
    )


# =============================================================================
# ESP32 Dual-IMU Live Transport & Validation Endpoints
# =============================================================================

class ValidatePacketRequest(BaseModel):
    packet_format: str = "json"  # "json", "csv", or "hex"
    payload: str


@app.get("/api/esp32/status")
def api_esp32_status() -> Dict[str, Any]:
    """Report live hardware connection state and diagnostic metrics."""
    ok, stall_err = esp32_validator.check_keepalive()
    status = esp32_validator.get_status_report()
    status["stall_warning"] = stall_err
    return {"status": "ok", "diagnostics": status}


@app.post("/api/esp32/validate")
def api_esp32_validate(req: ValidatePacketRequest) -> Dict[str, Any]:
    """Validate incoming live sensor packets and enforce dual-IMU health."""
    if req.packet_format == "json":
        res = esp32_validator.validate_json_packet(req.payload)
    elif req.packet_format == "csv":
        res = esp32_validator.validate_csv_line(req.payload)
    elif req.packet_format == "hex":
        try:
            raw_bytes = bytes.fromhex(req.payload.strip())
            res = esp32_validator.validate_binary_packet(raw_bytes)
        except Exception as exc:
            return {
                "status": "error",
                "is_valid": False,
                "error": f"Hex decoding failed: {exc}",
                "state": ConnectionState.INVALID_SENSOR_DATA.value,
            }
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported packet format: {req.packet_format}")

    return {
        "status": "ok" if res.is_valid else "invalid",
        "is_valid": res.is_valid,
        "state": res.state.value,
        "both_imus_active": res.both_imus_active,
        "rua_status": res.rua_status.value,
        "rla_status": res.rla_status.value,
        "timestamp_ms": res.timestamp_ms,
        "seq_num": res.seq_num,
        "channels": res.channels,
        "warning": res.warning_message,
        "error": res.error_message,
    }


@app.post("/api/esp32/reset")
def api_esp32_reset() -> Dict[str, Any]:
    """Reset validator to Disconnected state."""
    esp32_validator.reset_to_disconnected()
    return {"status": "ok", "state": ConnectionState.DISCONNECTED.value}


