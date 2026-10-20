"""Tests for the interactive demo FastAPI backend and inference engine."""

from __future__ import annotations

from pathlib import Path
import pytest

from demo.backend.app import (
    SampleAnalysisRequest,
    health_check,
    get_overview,
    list_samples,
    analyze_sample,
    serve_dashboard,
)
from demo.backend.engine import RehabilitationEngine

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_health_endpoint():
    data = health_check()
    assert data["status"] == "ok"
    assert "TCN" in data["supported_models"]
    assert "1D_CNN" in data["supported_models"]
    assert "LSTM" in data["supported_models"]
    assert "GRU" in data["supported_models"]


def test_overview_endpoint():
    data = get_overview()
    assert "manifest" in data
    assert "summary" in data
    assert data["samples_count"] >= 8


def test_samples_endpoint():
    data = list_samples()
    assert "samples" in data
    assert len(data["samples"]) >= 8


def test_analyze_sample_endpoint():
    samples_data = list_samples()
    first_sample = samples_data["samples"][0]
    sample_id = first_sample["sample_id"]

    req = SampleAnalysisRequest(sample_id=sample_id, selected_model="COMPARE_ALL")
    res_data = analyze_sample(req)

    assert res_data["status"] == "ok"
    assert "model_predictions" in res_data
    assert set(res_data["model_predictions"].keys()) == {"TCN", "1D_CNN", "LSTM", "GRU"}
    assert "comparison_tables" in res_data
    assert "task1_elbow_flexion" in res_data["comparison_tables"]
    assert len(res_data["comparison_tables"]["task1_elbow_flexion"]) == 4

    # Verify comprehensive kinematic outputs
    tcn_t1 = res_data["model_predictions"]["TCN"]["task1_elbow_flexion"]
    assert "current_angle_deg" in tcn_t1
    assert "angular_velocity_deg_s" in tcn_t1
    assert len(tcn_t1["angular_velocity_deg_s"]) == len(tcn_t1["predicted_trajectory"])
    assert "movement_smoothness_sparc" in tcn_t1
    assert "repetition_cycles" in tcn_t1
    assert "peak_flexion_velocity_deg_s" in tcn_t1
    assert "peak_extension_velocity_deg_s" in tcn_t1

    # Verify Stage 8: Final Clinical Assessment synthesis
    assert "final_clinical_assessment" in res_data
    fa = res_data["final_clinical_assessment"]
    assert "rom_status" in fa
    assert "smoothness_status" in fa
    assert "compensatory_risk" in fa
    assert "actionable_recommendations" in fa
    assert isinstance(fa["actionable_recommendations"], list)
    assert len(fa["actionable_recommendations"]) > 0
    assert "clinical_disclaimer" in fa
    assert "not a substitute" in fa["clinical_disclaimer"].lower()


def test_unlabeled_sample_does_not_fabricate_metrics():
    req = SampleAnalysisRequest(sample_id="sensor_only_unlabeled_recording", selected_model="TCN")
    res_data = analyze_sample(req)
    assert res_data["status"] == "ok"
    assert res_data["ground_truth_metadata"]["has_elbow_flex_gt"] is False
    tcn_t1 = res_data["model_predictions"]["TCN"]["task1_elbow_flexion"]
    assert "sample_mae_deg" not in tcn_t1
    assert "sample_rmse_deg" not in tcn_t1
    assert "sample_r2" not in tcn_t1


def test_frontend_serving():
    html_res = serve_dashboard()
    assert html_res.status_code == 200
    assert "AI Elbow Rehabilitation Assistant" in html_res.body.decode("utf-8")


def test_longitudinal_endpoints():
    from demo.backend.app import (
        api_list_participants,
        api_list_sessions,
        api_participant_summary,
        api_save_session,
        api_get_session,
        api_delete_session,
        api_export_csv,
        api_export_json,
        SaveSessionRequest,
    )

    # 1. Participants listing
    pts_res = api_list_participants()
    assert pts_res["status"] == "ok"
    assert "P012" in pts_res["participants"]

    # 2. Query sessions
    sess_res = api_list_sessions(participant_id="P012")
    assert sess_res["status"] == "ok"
    assert sess_res["count"] >= 5

    # 3. Longitudinal summary and descriptive trend estimation
    sum_res = api_participant_summary(participant_id="P012", target_rom_deg=130.0)
    assert sum_res["status"] == "ok"
    summary = sum_res["summary"]
    assert summary["participant_id"] == "P012"
    assert summary["baseline_elbow_rom_deg"] == 62.4
    assert summary["latest_elbow_rom_deg"] == 125.6
    assert summary["delta_elbow_rom_deg"] > 0
    assert summary["linear_trend"] is not None
    assert summary["linear_trend"]["slope_deg_per_session"] > 0
    assert "disclaimer" in summary
    assert "biological healing" in summary["disclaimer"].lower()

    # 4. Save and delete session
    new_sess = SaveSessionRequest(
        participant_id="TEST_PT_99",
        exercise_name="Elbow Flexion-Extension",
        elbow_rom_deg=88.5,
        max_elbow_angle_deg=98.0,
        min_elbow_angle_deg=9.5,
        repetition_count=7,
        movement_smoothness_sparc=-2.6,
        movement_quality_class="Correct Execution",
        notes="Automated test session",
    )
    save_res = api_save_session(new_sess)
    assert save_res["status"] == "ok"
    new_id = save_res["session_id"]

    fetched = api_get_session(new_id)
    assert fetched["status"] == "ok"
    assert fetched["session"]["participant_id"] == "TEST_PT_99"
    assert fetched["session"]["elbow_rom_deg"] == 88.5

    del_res = api_delete_session(new_id)
    assert del_res["status"] == "ok"

    # 5. Export CSV & JSON
    csv_res = api_export_csv(participant_id="P012")
    assert "session_id,timestamp,participant_id" in csv_res.body.decode("utf-8")
    assert "P012" in csv_res.body.decode("utf-8")

    json_res = api_export_json(participant_id="P012")
    import json
    parsed = json.loads(json_res.body.decode("utf-8"))
    assert parsed["status"] == "ok"
    assert parsed["session_count"] >= 5

