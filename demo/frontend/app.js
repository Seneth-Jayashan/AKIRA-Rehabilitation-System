/* Stage 4 Multi-Task Elbow Rehabilitation AI Dashboard Client */

const MODEL_COLORS = {
  TCN: "#1d4ed8",
  "1D_CNN": "#0f766e",
  LSTM: "#b45309",
  GRU: "#7e22ce",
  GROUND_TRUTH: "#0f172a",
};

let samplesList = [];

document.addEventListener("DOMContentLoaded", () => {
  initDashboard();
});

async function initDashboard() {
  try {
    const res = await fetch("/api/samples").then((r) => r.json());
    samplesList = res.samples || [];

    const select = document.getElementById("sample-select");
    if (select) {
      select.innerHTML = "";
      samplesList.forEach((s) => {
        const opt = document.createElement("option");
        opt.value = s.sample_id;
        const gtBadge = s.has_ground_truth ? "✓ [GT]" : "⚠ [Sensor Only]";
        opt.textContent = `${gtBadge} ${s.exercise_name} (${s.source_dataset}, ${s.subject_id}, ${s.duration_sec}s)`;
        select.appendChild(opt);
      });
      if (samplesList.length > 0) {
        select.value = samplesList[0].sample_id;
      }
    }

    bindEventListeners();
    await initLongitudinalModule();
    // Auto run on first sample
    if (samplesList.length > 0) {
      await runAnalysis();
    }
  } catch (err) {
    console.error("Failed to initialize dashboard:", err);
    updateStatus("Initialization error: " + err.message, true);
  }
}

function bindEventListeners() {
  const btnRun = document.getElementById("btn-run-analysis");
  const btnDownload = document.getElementById("btn-download-sample");
  const sampleSelect = document.getElementById("sample-select");
  const fileInput = document.getElementById("csv-file-input");

  if (btnRun) {
    btnRun.addEventListener("click", () => runAnalysis());
  }

  if (btnDownload) {
    btnDownload.addEventListener("click", () => {
      const sId = sampleSelect ? sampleSelect.value : null;
      if (sId) {
        window.open(`/api/samples/${encodeURIComponent(sId)}/download`, "_blank");
      }
    });
  }

  if (fileInput) {
    fileInput.addEventListener("change", () => {
      if (fileInput.files.length > 0) {
        updateStatus(`Custom file selected: ${fileInput.files[0].name}. Click Run Complete Analysis.`);
      }
    });
  }

  // Longitudinal tracking event listeners
  const btnToggleSave = document.getElementById("btn-toggle-save-session");
  const saveBar = document.getElementById("save-session-bar");
  if (btnToggleSave && saveBar) {
    btnToggleSave.addEventListener("click", () => {
      saveBar.classList.toggle("hidden");
    });
  }

  const btnConfirmSave = document.getElementById("btn-confirm-save-session");
  if (btnConfirmSave) {
    btnConfirmSave.addEventListener("click", () => handleSaveCurrentSession());
  }

  const btnCancelSave = document.getElementById("btn-cancel-save-session");
  if (btnCancelSave && saveBar) {
    btnCancelSave.addEventListener("click", () => saveBar.classList.add("hidden"));
  }

  const btnRefreshLong = document.getElementById("btn-refresh-longitudinal");
  if (btnRefreshLong) {
    btnRefreshLong.addEventListener("click", () => loadLongitudinalData());
  }

  const selPatient = document.getElementById("select-patient-filter");
  if (selPatient) {
    selPatient.addEventListener("change", () => loadLongitudinalData());
  }

  const btnExpCsv = document.getElementById("btn-export-csv");
  if (btnExpCsv) {
    btnExpCsv.addEventListener("click", () => {
      const pid = document.getElementById("select-patient-filter")?.value || "ALL";
      const sDate = document.getElementById("input-date-start")?.value || "";
      const eDate = document.getElementById("input-date-end")?.value || "";
      window.open(`/api/longitudinal/export/csv?participant_id=${encodeURIComponent(pid)}&start_date=${encodeURIComponent(sDate)}&end_date=${encodeURIComponent(eDate)}`, "_blank");
    });
  }

  const btnExpJson = document.getElementById("btn-export-json");
  if (btnExpJson) {
    btnExpJson.addEventListener("click", () => {
      const pid = document.getElementById("select-patient-filter")?.value || "ALL";
      const sDate = document.getElementById("input-date-start")?.value || "";
      const eDate = document.getElementById("input-date-end")?.value || "";
      window.open(`/api/longitudinal/export/json?participant_id=${encodeURIComponent(pid)}&start_date=${encodeURIComponent(sDate)}&end_date=${encodeURIComponent(eDate)}`, "_blank");
    });
  }

  // ESP32 Verified Hardware Transport & Simulation Event Listeners
  const btnConnectSerial = document.getElementById("btn-esp32-connect-serial");
  const btnConnectBle = document.getElementById("btn-esp32-connect-ble");
  const btnDisconnectEsp = document.getElementById("btn-esp32-disconnect");
  const btnSimReplay = document.getElementById("btn-esp32-sim-replay");
  const btnRecordEsp = document.getElementById("btn-esp32-record");
  const btnSaveCsvEsp = document.getElementById("btn-esp32-save-csv");
  const btnAnalyzeStreamEsp = document.getElementById("btn-esp32-analyze-stream");

  if (btnConnectSerial) {
    btnConnectSerial.addEventListener("click", () => connectEsp32Serial());
  }
  if (btnConnectBle) {
    btnConnectBle.addEventListener("click", () => connectEsp32Bluetooth());
  }
  if (btnDisconnectEsp) {
    btnDisconnectEsp.addEventListener("click", () => disconnectEsp32Device());
  }
  if (btnSimReplay) {
    btnSimReplay.addEventListener("click", () => toggleSimulationReplay());
  }
  if (btnRecordEsp) {
    btnRecordEsp.addEventListener("click", () => toggleEsp32Recording());
  }
  if (btnSaveCsvEsp) {
    btnSaveCsvEsp.addEventListener("click", () => saveEsp32RecordedCsv());
  }
  if (btnAnalyzeStreamEsp) {
    btnAnalyzeStreamEsp.addEventListener("click", () => analyzeRecordedLiveStream());
  }
}

function updateStatus(text, isError = false) {
  const el = document.getElementById("analysis-status-text");
  if (el) {
    el.textContent = text;
    el.style.color = isError ? "#b91c1c" : "#1e40af";
  }
}

async function runAnalysis() {
  const modelSelect = document.getElementById("model-select");
  const sampleSelect = document.getElementById("sample-select");
  const fileInput = document.getElementById("csv-file-input");
  const spinner = document.getElementById("analysis-spinner");

  if (!modelSelect) return;

  const selectedModel = modelSelect.value;
  const hasUpload = fileInput && fileInput.files && fileInput.files.length > 0;

  if (spinner) spinner.classList.remove("hidden");
  updateStatus("Running complete multi-task analysis across all models...");

  try {
    let res;
    if (hasUpload) {
      const file = fileInput.files[0];
      const formData = new FormData();
      formData.append("file", file);
      formData.append("selected_model", selectedModel);

      const r = await fetch("/api/analyze/upload", {
        method: "POST",
        body: formData,
      });
      if (!r.ok) throw new Error(await r.text());
      res = await r.json();
    } else {
      const sampleId = sampleSelect ? sampleSelect.value : null;
      if (!sampleId) throw new Error("Please select a sample CSV.");

      const r = await fetch("/api/analyze/sample", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sample_id: sampleId,
          selected_model: selectedModel,
        }),
      });
      if (!r.ok) throw new Error(await r.text());
      res = await r.json();
    }

    renderAnalysisResults(res);
    updateStatus(`Analysis complete! Processed ${res.num_timesteps} frames (${res.duration_sec}s at 100 Hz).`);
  } catch (err) {
    console.error("Analysis failed:", err);
    updateStatus("Analysis error: " + err.message, true);
  } finally {
    if (spinner) spinner.classList.add("hidden");
  }
}

function renderAnalysisResults(data) {
  window._lastAnalysisData = data;
  if (data.sample_metadata && data.sample_metadata.subject_id) {
    const ptInput = document.getElementById("input-session-patient-id");
    if (ptInput && data.sample_metadata.subject_id !== "Unknown") {
      ptInput.value = data.sample_metadata.subject_id.toUpperCase();
    }
  }
  renderSensorSignals(data);
  renderCsvPreview(data);
  renderTask1Flexion(data);
  renderTask2Prosup(data);
  renderTask3Exercise(data);
  renderTask4Quality(data);
  renderComparisonTables(data);
  renderFinalAssessment(data);
}

/* 1. SENSOR SIGNALS */
function renderSensorSignals(data) {
  const cUA = document.getElementById("canvas-ua-signals");
  const cFA = document.getElementById("canvas-fa-signals");
  const badge = document.getElementById("signal-info-badge");

  if (badge) {
    badge.textContent = `${data.filename} • ${data.duration_sec}s (${data.num_timesteps} frames @ 100 Hz)`;
  }

  const sigs = data.sensor_signals || {};
  const t = data.time_axis || [];

  if (cUA && sigs.RUA_Ax) {
    drawMultiLineChart(
      cUA,
      t,
      [
        { label: "Acc X", values: sigs.RUA_Ax, color: "#ef4444" },
        { label: "Acc Y", values: sigs.RUA_Ay, color: "#10b981" },
        { label: "Acc Z", values: sigs.RUA_Az, color: "#3b82f6" },
        { label: "Gyr X", values: sigs.RUA_Gx, color: "#f97316" },
        { label: "Gyr Y", values: sigs.RUA_Gy, color: "#8b5cf6" },
        { label: "Gyr Z", values: sigs.RUA_Gz, color: "#06b6d4" },
      ],
      "Time (s)",
      "Upper Arm (m/s² & rad/s)"
    );
  }

  if (cFA && sigs.RLA_Ax) {
    drawMultiLineChart(
      cFA,
      t,
      [
        { label: "Acc X", values: sigs.RLA_Ax, color: "#ef4444" },
        { label: "Acc Y", values: sigs.RLA_Ay, color: "#10b981" },
        { label: "Acc Z", values: sigs.RLA_Az, color: "#3b82f6" },
        { label: "Gyr X", values: sigs.RLA_Gx, color: "#f97316" },
        { label: "Gyr Y", values: sigs.RLA_Gy, color: "#8b5cf6" },
        { label: "Gyr Z", values: sigs.RLA_Gz, color: "#06b6d4" },
      ],
      "Time (s)",
      "Forearm (m/s² & rad/s)"
    );
  }
}

/* CSV TABLE PREVIEW */
function renderCsvPreview(data) {
  const container = document.getElementById("csv-preview-table-container");
  if (!container) return;

  const sigs = data.sensor_signals || {};
  const cols = Object.keys(sigs);
  if (!cols.length) return;

  const nRows = Math.min(8, data.num_timesteps);
  let html = '<table class="data-table"><thead><tr><th>Time (s)</th>';
  cols.forEach((c) => {
    html += `<th>${c}</th>`;
  });
  html += "</tr></thead><tbody>";

  for (let r = 0; r < nRows; r++) {
    html += `<tr><td>${data.time_axis[r]}</td>`;
    cols.forEach((c) => {
      html += `<td>${sigs[c][r]}</td>`;
    });
    html += "</tr>";
  }
  html += "</tbody></table>";
  container.innerHTML = html;
}

/* 2. TASK 1: ELBOW FLEXION */
function renderTask1Flexion(data) {
  const canvasTraj = document.getElementById("canvas-flexion-trajectory");
  const canvasVel = document.getElementById("canvas-flexion-velocity");
  const kpiGrid = document.getElementById("task1-kpi-grid");
  const repsBox = document.getElementById("task1-reps-table-box");
  const pillsBox = document.getElementById("task1-metrics-pills");
  const gtBadge = document.getElementById("task1-gt-status-badge");

  if (!canvasTraj) return;

  const hasGt = data.ground_truth_metadata && data.ground_truth_metadata.has_elbow_flex_gt;
  if (gtBadge) {
    gtBadge.textContent = hasGt ? "✓ Ground Truth Available" : "⚠ Reference Data Unavailable (Unlabeled)";
    gtBadge.style.background = hasGt ? "#dcfce7" : "#fef3c7";
    gtBadge.style.color = hasGt ? "#15803d" : "#92400e";
  }

  const preds = data.model_predictions || {};
  const modelNames = Object.keys(preds);
  const selectedModel = data.selected_model;
  const primaryModelName = (selectedModel && preds[selectedModel]) ? selectedModel : (preds["TCN"] ? "TCN" : modelNames[0]);
  const primaryT1 = preds[primaryModelName] ? preds[primaryModelName].task1_elbow_flexion : null;

  // 1. Render KPI Grid
  if (kpiGrid && primaryT1) {
    const curAngle = primaryT1.current_angle_deg !== undefined ? primaryT1.current_angle_deg.toFixed(1) : "N/A";
    const trialRom = primaryT1.full_trial_rom_deg !== undefined ? primaryT1.full_trial_rom_deg.toFixed(1) : "N/A";
    const repMeanRom = primaryT1.repetition_mean_rom_deg !== undefined && primaryT1.repetition_mean_rom_deg !== null
      ? primaryT1.repetition_mean_rom_deg.toFixed(1) + "°" : "N/A";
    const repCount = primaryT1.num_repetitions_detected !== undefined ? primaryT1.num_repetitions_detected : 0;
    const peakFlexVel = primaryT1.peak_flexion_velocity_deg_s !== undefined ? primaryT1.peak_flexion_velocity_deg_s.toFixed(1) : "0.0";
    const peakExtVel = primaryT1.peak_extension_velocity_deg_s !== undefined ? primaryT1.peak_extension_velocity_deg_s.toFixed(1) : "0.0";
    const sparcVal = primaryT1.movement_smoothness_sparc !== undefined && primaryT1.movement_smoothness_sparc !== null
      ? primaryT1.movement_smoothness_sparc.toFixed(2) : "N/A";
    let sparcText = "No velocity data";
    if (primaryT1.movement_smoothness_sparc !== undefined && primaryT1.movement_smoothness_sparc !== null) {
      if (primaryT1.movement_smoothness_sparc >= -2.5) sparcText = "Smooth / Coordinated";
      else if (primaryT1.movement_smoothness_sparc >= -3.5) sparcText = "Moderate Sub-movements";
      else sparcText = "Arrested / Jerky Execution";
    }

    let gtCardVal = "N/A";
    let gtCardSub = "Reference data unavailable";
    if (hasGt) {
      if (primaryT1.sample_mae_deg !== undefined && primaryT1.sample_mae_deg !== null) {
        gtCardVal = `MAE: ${primaryT1.sample_mae_deg.toFixed(2)}°`;
        const r2Str = primaryT1.sample_r2 !== undefined && primaryT1.sample_r2 !== null ? `R²: ${primaryT1.sample_r2.toFixed(3)} • ` : "";
        gtCardSub = `${r2Str}RMSE: ${primaryT1.sample_rmse_deg?.toFixed(2) || '0.00'}°`;
      }
    }

    kpiGrid.innerHTML = `
      <div class="kpi-card hero-primary">
        <span class="kpi-title">Current Angle (${primaryModelName})</span>
        <span class="kpi-val">${curAngle}°</span>
        <span class="kpi-sub">Range: ${primaryT1.min_angle_deg?.toFixed(1) || '0'}° – ${primaryT1.max_angle_deg?.toFixed(1) || '0'}°</span>
      </div>
      <div class="kpi-card">
        <span class="kpi-title">Trial ROM (Max - Min)</span>
        <span class="kpi-val">${trialRom}°</span>
        <span class="kpi-sub">Mean Rep ROM: ${repMeanRom}</span>
      </div>
      <div class="kpi-card">
        <span class="kpi-title">Repetitions Segmented</span>
        <span class="kpi-val">${repCount}</span>
        <span class="kpi-sub">${primaryT1.repetition_cycles?.length || 0} discrete cycles detected</span>
      </div>
      <div class="kpi-card">
        <span class="kpi-title">Angular Velocity</span>
        <span class="kpi-val">${Math.max(Math.abs(parseFloat(peakFlexVel)), Math.abs(parseFloat(peakExtVel))).toFixed(1)} °/s</span>
        <span class="kpi-sub">Flex: +${peakFlexVel}°/s • Ext: ${peakExtVel}°/s</span>
      </div>
      <div class="kpi-card">
        <span class="kpi-title">Smoothness (SPARC)</span>
        <span class="kpi-val">${sparcVal}</span>
        <span class="kpi-sub">${sparcText}</span>
      </div>
      <div class="kpi-card">
        <span class="kpi-title">Ground Truth Verification</span>
        <span class="kpi-val">${gtCardVal}</span>
        <span class="kpi-sub">${gtCardSub}</span>
      </div>
    `;
  }

  // 2. Trajectory Chart
  const trajSeries = [];
  const t = data.time_axis || [];

  if (hasGt && Array.isArray(data.ground_truth_metadata.true_elbow_flex_traj)) {
    trajSeries.push({
      label: "Reference Ground Truth (°)",
      values: data.ground_truth_metadata.true_elbow_flex_traj,
      color: MODEL_COLORS.GROUND_TRUTH,
      lineWidth: 3.0,
    });
  }

  modelNames.forEach((mName) => {
    const t1 = preds[mName].task1_elbow_flexion;
    if (t1 && Array.isArray(t1.predicted_trajectory)) {
      trajSeries.push({
        label: `${mName} Estimated (°)${mName === primaryModelName ? ' [Active]' : ''}`,
        values: t1.predicted_trajectory,
        color: MODEL_COLORS[mName] || "#2563eb",
        lineWidth: mName === primaryModelName ? 2.5 : 1.5,
      });
    }
  });

  drawMultiLineChart(canvasTraj, t, trajSeries, "Time (s)", "Elbow Flexion Angle (°)");

  // 3. Continuous Velocity Chart
  if (canvasVel) {
    const velSeries = [];
    if (hasGt && Array.isArray(data.ground_truth_metadata.true_elbow_velocity_deg_s)) {
      velSeries.push({
        label: "Reference Angular Velocity (°/s)",
        values: data.ground_truth_metadata.true_elbow_velocity_deg_s,
        color: MODEL_COLORS.GROUND_TRUTH,
        lineWidth: 2.5,
      });
    }

    modelNames.forEach((mName) => {
      const t1 = preds[mName].task1_elbow_flexion;
      if (t1 && Array.isArray(t1.angular_velocity_deg_s)) {
        velSeries.push({
          label: `${mName} Velocity (°/s)`,
          values: t1.angular_velocity_deg_s,
          color: MODEL_COLORS[mName] || "#2563eb",
          lineWidth: mName === primaryModelName ? 2.0 : 1.2,
        });
      }
    });

    drawMultiLineChart(canvasVel, t, velSeries, "Time (s)", "Angular Velocity (°/s)");
  }

  // 4. Repetition Breakdown Table
  if (repsBox && primaryT1) {
    const reps = primaryT1.repetition_cycles || [];
    if (reps.length > 0) {
      let rHtml = `
        <table class="reps-table">
          <thead>
            <tr>
              <th>Rep #</th>
              <th>Start Time (s)</th>
              <th>Peak Time (s)</th>
              <th>End Time (s)</th>
              <th>Duration (s)</th>
              <th>Min Angle (°)</th>
              <th>Max (Peak) (°)</th>
              <th>Cycle ROM (°)</th>
            </tr>
          </thead>
          <tbody>
      `;
      reps.forEach((rep) => {
        rHtml += `
          <tr>
            <td style="font-weight:700; color:#1e40af;">Cycle ${rep.rep_index}</td>
            <td>${rep.start_time_s.toFixed(2)}</td>
            <td>${rep.peak_time_s.toFixed(2)}</td>
            <td>${rep.end_time_s.toFixed(2)}</td>
            <td>${((rep.duration_sec ?? rep.duration_s) || 0).toFixed(2)} s</td>
            <td>${rep.min_angle_deg.toFixed(1)}°</td>
            <td style="font-weight:600;">${rep.max_angle_deg.toFixed(1)}°</td>
            <td style="font-weight:700; color:#15803d;">${((rep.rom_deg ?? rep.rep_rom_deg) || 0).toFixed(1)}°</td>
          </tr>
        `;
      });
      rHtml += `</tbody></table>`;
      repsBox.innerHTML = rHtml;
    } else {
      repsBox.innerHTML = `<p style="padding:0.6rem 0.2rem; color:#64748b; font-size:0.82rem; margin:0;">No individual cyclical repetitions segmented in this recording window.</p>`;
    }
  }

  // 5. Cross-model summary pills
  if (pillsBox) {
    pillsBox.innerHTML = "";
    if (hasGt) {
      const gtTraj = data.ground_truth_metadata.true_elbow_flex_traj;
      const trueRom = (Math.max(...gtTraj) - Math.min(...gtTraj)).toFixed(1);
      const trueSparc = data.ground_truth_metadata.true_sparc !== undefined && data.ground_truth_metadata.true_sparc !== null
        ? data.ground_truth_metadata.true_sparc.toFixed(2) : "N/A";
      pillsBox.innerHTML += `
        <div class="metric-pill">
          <span class="metric-pill-label">Reference Ground Truth</span>
          <span class="metric-pill-value">ROM: ${trueRom}° • SPARC: ${trueSparc}</span>
        </div>
      `;
    }

    modelNames.forEach((mName) => {
      const t1 = preds[mName].task1_elbow_flexion;
      const isWinner = mName === "TCN";
      let text = `ROM: ${t1.full_trial_rom_deg}°`;
      if (t1.sample_mae_deg !== undefined && t1.sample_mae_deg !== null) {
        text += ` • MAE: ${t1.sample_mae_deg}°`;
      }
      if (t1.sample_r2 !== undefined && t1.sample_r2 !== null) {
        text += ` • R²: ${t1.sample_r2.toFixed(3)}`;
      }
      if (t1.movement_smoothness_sparc !== undefined && t1.movement_smoothness_sparc !== null) {
        text += ` • SPARC: ${t1.movement_smoothness_sparc.toFixed(2)}`;
      }

      pillsBox.innerHTML += `
        <div class="metric-pill ${isWinner ? 'winner' : ''}">
          <span class="metric-pill-label">${mName} Kinematics</span>
          <span class="metric-pill-value">${text}</span>
        </div>
      `;
    });
  }
}

/* 3. TASK 2: FOREARM PROSUP */
function renderTask2Prosup(data) {
  const canvasTraj = document.getElementById("canvas-prosup-trajectory");
  const canvasVel = document.getElementById("canvas-prosup-velocity");
  const kpiGrid = document.getElementById("task2-kpi-grid");
  const repsBox = document.getElementById("task2-reps-table-box");
  const pillsBox = document.getElementById("task2-metrics-pills");
  const gtBadge = document.getElementById("task2-gt-status-badge");

  if (!canvasTraj) return;

  const hasGt = data.ground_truth_metadata && data.ground_truth_metadata.has_forearm_prosup_gt;
  if (gtBadge) {
    gtBadge.textContent = hasGt ? "✓ Ground Truth Available" : "⚠ Reference Data Unavailable (Unlabeled)";
    gtBadge.style.background = hasGt ? "#dcfce7" : "#fef3c7";
    gtBadge.style.color = hasGt ? "#15803d" : "#92400e";
  }

  const preds = data.model_predictions || {};
  const modelNames = Object.keys(preds);
  const selectedModel = data.selected_model;
  const primaryModelName = (selectedModel && preds[selectedModel]) ? selectedModel : (preds["TCN"] ? "TCN" : modelNames[0]);
  const primaryT2 = preds[primaryModelName] ? preds[primaryModelName].task2_forearm_prosup : null;

  // 1. Render KPI Grid
  if (kpiGrid && primaryT2) {
    const curAngle = primaryT2.current_angle_deg !== undefined ? primaryT2.current_angle_deg.toFixed(1) : "N/A";
    const trialRom = primaryT2.full_trial_rom_deg !== undefined ? primaryT2.full_trial_rom_deg.toFixed(1) : "N/A";
    const repCount = primaryT2.num_repetitions_detected !== undefined ? primaryT2.num_repetitions_detected : 0;
    const peakVel = Math.max(
      Math.abs(primaryT2.peak_flexion_velocity_deg_s || 0),
      Math.abs(primaryT2.peak_extension_velocity_deg_s || 0)
    ).toFixed(1);
    const sparcVal = primaryT2.movement_smoothness_sparc !== undefined && primaryT2.movement_smoothness_sparc !== null
      ? primaryT2.movement_smoothness_sparc.toFixed(2) : "N/A";
    let sparcText = "No velocity data";
    if (primaryT2.movement_smoothness_sparc !== undefined && primaryT2.movement_smoothness_sparc !== null) {
      if (primaryT2.movement_smoothness_sparc >= -2.5) sparcText = "Smooth / Coordinated";
      else if (primaryT2.movement_smoothness_sparc >= -3.5) sparcText = "Moderate Sub-movements";
      else sparcText = "Arrested / Jerky Execution";
    }

    let gtCardVal = "N/A";
    let gtCardSub = "Reference data unavailable";
    if (hasGt) {
      if (primaryT2.sample_mae_deg !== undefined && primaryT2.sample_mae_deg !== null) {
        gtCardVal = `MAE: ${primaryT2.sample_mae_deg.toFixed(2)}°`;
        const r2Str = primaryT2.sample_r2 !== undefined && primaryT2.sample_r2 !== null ? `R²: ${primaryT2.sample_r2.toFixed(3)} • ` : "";
        gtCardSub = `${r2Str}RMSE: ${primaryT2.sample_rmse_deg?.toFixed(2) || '0.00'}°`;
      }
    }

    kpiGrid.innerHTML = `
      <div class="kpi-card hero-primary">
        <span class="kpi-title">Current Rotation (${primaryModelName})</span>
        <span class="kpi-val">${curAngle}°</span>
        <span class="kpi-sub">Range: ${primaryT2.min_angle_deg?.toFixed(1) || '0'}° – ${primaryT2.max_angle_deg?.toFixed(1) || '0'}°</span>
      </div>
      <div class="kpi-card">
        <span class="kpi-title">Rotational ROM</span>
        <span class="kpi-val">${trialRom}°</span>
        <span class="kpi-sub">Total span across trial</span>
      </div>
      <div class="kpi-card">
        <span class="kpi-title">Rotational Cycles</span>
        <span class="kpi-val">${repCount}</span>
        <span class="kpi-sub">${primaryT2.repetition_cycles?.length || 0} discrete cycles</span>
      </div>
      <div class="kpi-card">
        <span class="kpi-title">Peak Rotational Velocity</span>
        <span class="kpi-val">${peakVel} °/s</span>
        <span class="kpi-sub">Peak angular rate of rotation</span>
      </div>
      <div class="kpi-card">
        <span class="kpi-title">Smoothness (SPARC)</span>
        <span class="kpi-val">${sparcVal}</span>
        <span class="kpi-sub">${sparcText}</span>
      </div>
      <div class="kpi-card">
        <span class="kpi-title">Ground Truth Verification</span>
        <span class="kpi-val">${gtCardVal}</span>
        <span class="kpi-sub">${gtCardSub}</span>
      </div>
    `;
  }

  // 2. Trajectory Chart
  const trajSeries = [];
  const t = data.time_axis || [];

  if (hasGt && Array.isArray(data.ground_truth_metadata.true_forearm_prosup_traj)) {
    trajSeries.push({
      label: "Reference Ground Truth (°)",
      values: data.ground_truth_metadata.true_forearm_prosup_traj,
      color: MODEL_COLORS.GROUND_TRUTH,
      lineWidth: 3.0,
    });
  }

  modelNames.forEach((mName) => {
    const t2 = preds[mName].task2_forearm_prosup;
    if (t2 && Array.isArray(t2.predicted_trajectory)) {
      trajSeries.push({
        label: `${mName} Estimated (°)${mName === primaryModelName ? ' [Active]' : ''}`,
        values: t2.predicted_trajectory,
        color: MODEL_COLORS[mName] || "#2563eb",
        lineWidth: mName === primaryModelName ? 2.5 : 1.5,
      });
    }
  });

  drawMultiLineChart(canvasTraj, t, trajSeries, "Time (s)", "Forearm Pronation/Supination Angle (°)");

  // 3. Continuous Velocity Chart
  if (canvasVel) {
    const velSeries = [];
    if (hasGt && Array.isArray(data.ground_truth_metadata.true_forearm_prosup_velocity_deg_s)) {
      velSeries.push({
        label: "Reference Angular Velocity (°/s)",
        values: data.ground_truth_metadata.true_forearm_prosup_velocity_deg_s,
        color: MODEL_COLORS.GROUND_TRUTH,
        lineWidth: 2.5,
      });
    }

    modelNames.forEach((mName) => {
      const t2 = preds[mName].task2_forearm_prosup;
      if (t2 && Array.isArray(t2.angular_velocity_deg_s)) {
        velSeries.push({
          label: `${mName} Velocity (°/s)`,
          values: t2.angular_velocity_deg_s,
          color: MODEL_COLORS[mName] || "#2563eb",
          lineWidth: mName === primaryModelName ? 2.0 : 1.2,
        });
      }
    });

    drawMultiLineChart(canvasVel, t, velSeries, "Time (s)", "Angular Velocity (°/s)");
  }

  // 4. Repetition Breakdown Table
  if (repsBox && primaryT2) {
    const reps = primaryT2.repetition_cycles || [];
    if (reps.length > 0) {
      let rHtml = `
        <table class="reps-table">
          <thead>
            <tr>
              <th>Cycle #</th>
              <th>Start Time (s)</th>
              <th>Peak Time (s)</th>
              <th>End Time (s)</th>
              <th>Duration (s)</th>
              <th>Min Angle (°)</th>
              <th>Max (Peak) (°)</th>
              <th>Cycle ROM (°)</th>
            </tr>
          </thead>
          <tbody>
      `;
      reps.forEach((rep) => {
        rHtml += `
          <tr>
            <td style="font-weight:700; color:#1e40af;">Cycle ${rep.rep_index}</td>
            <td>${rep.start_time_s.toFixed(2)}</td>
            <td>${rep.peak_time_s.toFixed(2)}</td>
            <td>${rep.end_time_s.toFixed(2)}</td>
            <td>${((rep.duration_sec ?? rep.duration_s) || 0).toFixed(2)} s</td>
            <td>${rep.min_angle_deg.toFixed(1)}°</td>
            <td style="font-weight:600;">${rep.max_angle_deg.toFixed(1)}°</td>
            <td style="font-weight:700; color:#15803d;">${((rep.rom_deg ?? rep.rep_rom_deg) || 0).toFixed(1)}°</td>
          </tr>
        `;
      });
      rHtml += `</tbody></table>`;
      repsBox.innerHTML = rHtml;
    } else {
      repsBox.innerHTML = `<p style="padding:0.6rem 0.2rem; color:#64748b; font-size:0.82rem; margin:0;">No individual cyclical repetitions segmented in this recording window.</p>`;
    }
  }

  // 5. Cross-model summary pills
  if (pillsBox) {
    pillsBox.innerHTML = "";
    if (hasGt) {
      const gtTraj = data.ground_truth_metadata.true_forearm_prosup_traj;
      const trueRom = (Math.max(...gtTraj) - Math.min(...gtTraj)).toFixed(1);
      const trueSparc = data.ground_truth_metadata.true_prosup_sparc !== undefined && data.ground_truth_metadata.true_prosup_sparc !== null
        ? data.ground_truth_metadata.true_prosup_sparc.toFixed(2) : "N/A";
      pillsBox.innerHTML += `
        <div class="metric-pill">
          <span class="metric-pill-label">Reference Ground Truth</span>
          <span class="metric-pill-value">ROM: ${trueRom}° • SPARC: ${trueSparc}</span>
        </div>
      `;
    }

    modelNames.forEach((mName) => {
      const t2 = preds[mName].task2_forearm_prosup;
      let text = `ROM: ${t2.full_trial_rom_deg}°`;
      if (t2.sample_mae_deg !== undefined && t2.sample_mae_deg !== null) {
        text += ` • MAE: ${t2.sample_mae_deg}°`;
      }
      if (t2.sample_r2 !== undefined && t2.sample_r2 !== null) {
        text += ` • R²: ${t2.sample_r2.toFixed(3)}`;
      }
      if (t2.movement_smoothness_sparc !== undefined && t2.movement_smoothness_sparc !== null) {
        text += ` • SPARC: ${t2.movement_smoothness_sparc.toFixed(2)}`;
      }

      pillsBox.innerHTML += `
        <div class="metric-pill">
          <span class="metric-pill-label">${mName} Kinematics</span>
          <span class="metric-pill-value">${text}</span>
        </div>
      `;
    });
  }
}

/* 4. TASK 3: EXERCISE CLASSIFICATION */
function renderTask3Exercise(data) {
  const canvas = document.getElementById("canvas-exercise-probs");
  const pillsBox = document.getElementById("task3-metrics-pills");
  const gtBadge = document.getElementById("task3-gt-status-badge");

  if (!canvas || !pillsBox) return;

  const hasGt = data.ground_truth_metadata && data.ground_truth_metadata.has_exercise_class_gt;
  if (gtBadge) {
    gtBadge.textContent = hasGt ? "✓ Ground Truth Available" : "⚠ Unlabeled Recording (No GT)";
    gtBadge.style.background = hasGt ? "#dcfce7" : "#fef3c7";
    gtBadge.style.color = hasGt ? "#15803d" : "#92400e";
  }

  const preds = data.model_predictions || {};
  const modelNames = Object.keys(preds);
  const firstM = modelNames[0];
  const t3Data = firstM ? preds[firstM].task3_exercise_classification : null;
  const classes = (t3Data && t3Data.classes) ? t3Data.classes : [
    "Elbow Flexion-Extension",
    "Forearm Pronation-Supination",
    "Shoulder Flexion / Overhead Reach",
    "Shoulder Abduction / Scaption",
    "Shoulder Internal-External Rotation",
    "Crossbody Reach / Horizontal Adduction",
    "Functional Drinking / Reach-to-Grasp",
  ];

  const palette = [
    "#2563eb", "#059669", "#d97706", "#7c3aed", "#db2777", "#0891b2", "#ea580c", "#475569"
  ];

  const seriesList = classes.map((cName, cIdx) => {
    return {
      label: cName,
      values: modelNames.map((m) => {
        const probs = preds[m].task3_exercise_classification.class_probabilities || [];
        return Number((probs[cIdx] || 0) * 100);
      }),
      color: palette[cIdx % palette.length],
    };
  });

  drawGroupedBarChart(canvas, modelNames, seriesList, { decimals: 1, suffix: "%", yMax: 100 });

  pillsBox.innerHTML = "";
  if (hasGt) {
    pillsBox.innerHTML += `
      <div class="metric-pill">
        <span class="metric-pill-label">Ground-Truth Exercise</span>
        <span class="metric-pill-value">${data.ground_truth_metadata.true_exercise_name || 'N/A'}</span>
      </div>
    `;
  }

  modelNames.forEach((mName) => {
    const t3 = preds[mName].task3_exercise_classification;
    const isWinner = mName === "LSTM";
    const matchTag = t3.is_correct !== undefined ? (t3.is_correct ? " [✓ Match]" : " [✗ Mismatch]") : "";
    pillsBox.innerHTML += `
      <div class="metric-pill ${isWinner ? 'winner' : ''}">
        <span class="metric-pill-label">${mName} Prediction</span>
        <span class="metric-pill-value">${t3.predicted_class_name} (${t3.confidence_pct}%)${matchTag}</span>
      </div>
    `;
  });
}

/* 5. TASK 4: MOVEMENT QUALITY */
function renderTask4Quality(data) {
  const canvas = document.getElementById("canvas-quality-probs");
  const pillsBox = document.getElementById("task4-metrics-pills");
  const gtBadge = document.getElementById("task4-gt-status-badge");

  if (!canvas || !pillsBox) return;

  const hasGt = data.ground_truth_metadata && data.ground_truth_metadata.has_quality_class_gt;
  if (gtBadge) {
    gtBadge.textContent = hasGt ? "✓ Ground Truth Available" : "⚠ Unlabeled Recording (No GT)";
    gtBadge.style.background = hasGt ? "#dcfce7" : "#fef3c7";
    gtBadge.style.color = hasGt ? "#15803d" : "#92400e";
  }

  const preds = data.model_predictions || {};
  const modelNames = Object.keys(preds);
  const classes = ["Correct Execution", "Incorrect / Compensatory"];

  const seriesList = classes.map((cName, cIdx) => {
    const colors = ["#15803d", "#b91c1c"];
    return {
      label: cName,
      values: modelNames.map((m) => Number(preds[m].task4_quality_classification.quality_probabilities[cIdx] * 100)),
      color: colors[cIdx],
    };
  });

  drawGroupedBarChart(canvas, modelNames, seriesList, { decimals: 1, suffix: "%", yMax: 100 });

  pillsBox.innerHTML = "";
  if (hasGt) {
    pillsBox.innerHTML += `
      <div class="metric-pill">
        <span class="metric-pill-label">Ground-Truth Quality</span>
        <span class="metric-pill-value">${data.ground_truth_metadata.true_quality_name || 'N/A'}</span>
      </div>
    `;
  }

  modelNames.forEach((mName) => {
    const t4 = preds[mName].task4_quality_classification;
    const matchTag = t4.is_correct !== undefined ? (t4.is_correct ? " [✓ Correct]" : " [✗ Incorrect]") : "";
    pillsBox.innerHTML += `
      <div class="metric-pill">
        <span class="metric-pill-label">${mName} Prediction</span>
        <span class="metric-pill-value">${t4.predicted_quality_name} (${t4.confidence_pct}%)${matchTag}</span>
      </div>
    `;
  });
}

/* 6. COMPARISON TABLES */
function renderComparisonTables(data) {
  const tables = data.comparison_tables || {};

  // Task 1
  renderSingleComparisonTable("table-comp-task1", tables.task1_elbow_flexion || [], (r) => `
    <td style="font-weight:700; color:${MODEL_COLORS[r.model_name]};">${r.model_name}</td>
    <td>${r.sample_mae_deg !== null && r.sample_mae_deg !== undefined ? r.sample_mae_deg + '°' : 'N/A'}</td>
    <td>${r.sample_rmse_deg !== null && r.sample_rmse_deg !== undefined ? r.sample_rmse_deg + '°' : 'N/A'}</td>
    <td style="font-weight:600;">${r.predicted_rom_deg}°</td>
    <td>${r.rom_error_deg !== null && r.rom_error_deg !== undefined ? r.rom_error_deg + '°' : 'N/A'}</td>
    <td>${r.num_repetitions}</td>
    <td style="font-weight:700; color:#15803d;">${r.test_overall_mae_deg ? r.test_overall_mae_deg + '°' : 'N/A'}</td>
    <td>${r.test_overall_r2 || 'N/A'}</td>
    <td style="color:#64748b;">${r.val_mae_deg ? r.val_mae_deg + '°' : 'N/A'}</td>
    <td>${r.latency_ms ? r.latency_ms + ' ms' : 'N/A'}</td>
    <td>${r.parameters ? r.parameters.toLocaleString() : 'N/A'}</td>
    <td>${r.model_name === 'TCN' ? '<span class="badge" style="background:#dcfce7; color:#15803d; font-weight:700;">★ Best Model</span>' : '<span class="badge">Competitive</span>'}</td>
  `);

  // Task 2
  renderSingleComparisonTable("table-comp-task2", tables.task2_forearm_prosup || [], (r) => `
    <td style="font-weight:700; color:${MODEL_COLORS[r.model_name]};">${r.model_name}</td>
    <td>${r.sample_mae_deg !== null && r.sample_mae_deg !== undefined ? r.sample_mae_deg + '°' : 'N/A'}</td>
    <td>${r.sample_rmse_deg !== null && r.sample_rmse_deg !== undefined ? r.sample_rmse_deg + '°' : 'N/A'}</td>
    <td style="font-weight:600;">${r.predicted_rom_deg}°</td>
    <td>${r.rom_error_deg !== null && r.rom_error_deg !== undefined ? r.rom_error_deg + '°' : 'N/A'}</td>
    <td style="font-weight:700;">${r.test_overall_mae_deg ? r.test_overall_mae_deg + '°' : 'N/A'}</td>
    <td>${r.test_overall_r2 || 'N/A'}</td>
    <td style="color:#64748b;">${r.val_mae_deg ? r.val_mae_deg + '°' : 'N/A'}</td>
    <td>${r.latency_ms ? r.latency_ms + ' ms' : 'N/A'}</td>
    <td>${r.parameters ? r.parameters.toLocaleString() : 'N/A'}</td>
    <td><span class="badge">Validated</span></td>
  `);

  // Task 3
  renderSingleComparisonTable("table-comp-task3", tables.task3_exercise_classification || [], (r) => `
    <td style="font-weight:700; color:${MODEL_COLORS[r.model_name]};">${r.model_name}</td>
    <td style="font-weight:600;">${r.predicted_class}</td>
    <td>${r.confidence_pct}%</td>
    <td>${r.is_correct !== null && r.is_correct !== undefined ? (r.is_correct ? '<span class="badge" style="background:#dcfce7; color:#15803d;">✓ Match</span>' : '<span class="badge" style="background:#fee2e2; color:#b91c1c;">✗ Mismatch</span>') : 'N/A'}</td>
    <td style="font-weight:700;">${r.test_macro_f1 || 'N/A'}</td>
    <td>${r.test_accuracy ? (r.test_accuracy * 100).toFixed(1) + '%' : 'N/A'}</td>
    <td style="color:#64748b;">${r.val_macro_f1 || 'N/A'}</td>
    <td>${r.latency_ms ? r.latency_ms + ' ms' : 'N/A'}</td>
    <td>${r.parameters ? r.parameters.toLocaleString() : 'N/A'}</td>
    <td>${r.model_name === 'LSTM' ? '<span class="badge" style="background:#dcfce7; color:#15803d; font-weight:700;">★ Best Val F1</span>' : '<span class="badge">Competitive</span>'}</td>
  `);

  // Task 4
  renderSingleComparisonTable("table-comp-task4", tables.task4_quality_classification || [], (r) => `
    <td style="font-weight:700; color:${MODEL_COLORS[r.model_name]};">${r.model_name}</td>
    <td style="font-weight:600;">${r.predicted_quality}</td>
    <td>${r.confidence_pct}%</td>
    <td>${r.is_correct !== null && r.is_correct !== undefined ? (r.is_correct ? '<span class="badge" style="background:#dcfce7; color:#15803d;">✓ Match</span>' : '<span class="badge" style="background:#fee2e2; color:#b91c1c;">✗ Mismatch</span>') : 'N/A'}</td>
    <td style="font-weight:700;">${r.test_macro_f1 || 'N/A'}</td>
    <td>${r.test_accuracy ? (r.test_accuracy * 100).toFixed(1) + '%' : 'N/A'}</td>
    <td style="color:#64748b;">${r.val_macro_f1 || 'N/A'}</td>
    <td>${r.latency_ms ? r.latency_ms + ' ms' : 'N/A'}</td>
    <td>${r.parameters ? r.parameters.toLocaleString() : 'N/A'}</td>
    <td><span class="badge">Validated</span></td>
  `);
}

function renderSingleComparisonTable(tableId, rows, rowFormatter) {
  const table = document.getElementById(tableId);
  if (!table) return;
  const tbody = table.querySelector("tbody");
  if (!tbody) return;

  tbody.innerHTML = "";
  rows.forEach((r) => {
    const tr = document.createElement("tr");
    tr.innerHTML = rowFormatter(r);
    tbody.appendChild(tr);
  });
}

/* OFFLINE CANVAS DRAWING UTILITIES */
function drawMultiLineChart(canvas, xValues, seriesList, xLabel, yLabel) {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const padL = 50;
  const padR = 16;
  const padT = 28;
  const padB = 34;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  const validSeries = seriesList.filter((s) => Array.isArray(s.values) && s.values.length > 0);
  if (!validSeries.length || !xValues.length) return;

  const allY = validSeries.flatMap((s) => s.values).filter((v) => Number.isFinite(v));
  let yMin = Math.min(...allY);
  let yMax = Math.max(...allY);
  if (Math.abs(yMax - yMin) < 1e-4) {
    yMin -= 1;
    yMax += 1;
  } else {
    const margin = 0.08 * (yMax - yMin);
    yMin -= margin;
    yMax += margin;
  }

  const xMin = xValues[0];
  const xMax = xValues[xValues.length - 1] > xMin ? xValues[xValues.length - 1] : xMin + 1;

  ctx.strokeStyle = "#e2e8f0";
  ctx.fillStyle = "#64748b";
  ctx.font = "10.5px sans-serif";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const val = yMin + ((yMax - yMin) * i) / 4;
    const y = padT + plotH - ((val - yMin) / (yMax - yMin)) * plotH;
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(w - padR, y);
    ctx.stroke();
    ctx.textAlign = "right";
    ctx.fillText(val.toFixed(1), padL - 6, y + 3);
  }

  let legX = padL;
  validSeries.forEach((s) => {
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 2.4;
    ctx.beginPath();
    ctx.moveTo(legX, 13);
    ctx.lineTo(legX + 14, 13);
    ctx.stroke();
    ctx.fillStyle = "#0f172a";
    ctx.font = "600 10.5px sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(s.label, legX + 18, 17);
    legX += ctx.measureText(s.label).width + 24;
  });

  validSeries.forEach((s) => {
    ctx.strokeStyle = s.color;
    ctx.lineWidth = s.lineWidth || 1.6;
    ctx.beginPath();
    const len = Math.min(xValues.length, s.values.length);
    for (let i = 0; i < len; i++) {
      const px = padL + ((xValues[i] - xMin) / (xMax - xMin)) * plotW;
      const py = padT + plotH - ((s.values[i] - yMin) / (yMax - yMin)) * plotH;
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();
  });

  ctx.fillStyle = "#64748b";
  ctx.textAlign = "center";
  for (let i = 0; i <= 4; i++) {
    const xv = xMin + ((xMax - xMin) * i) / 4;
    const px = padL + (i / 4) * plotW;
    ctx.fillText(xv.toFixed(1) + "s", px, padT + plotH + 15);
  }
  ctx.fillText(`${xLabel} • ${yLabel}`, padL + plotW / 2, h - 4);
}

function drawGroupedBarChart(canvas, categories, seriesList, opts = {}) {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const padL = 48;
  const padR = 18;
  const padT = 30;
  const padB = 42;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  const yMax = opts.yMax || 100;

  ctx.strokeStyle = "#e2e8f0";
  ctx.fillStyle = "#64748b";
  ctx.font = "11px sans-serif";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const val = (yMax * i) / 4;
    const y = padT + plotH - (val / yMax) * plotH;
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(w - padR, y);
    ctx.stroke();
    ctx.textAlign = "right";
    ctx.fillText(val.toFixed(0) + (opts.suffix || ""), padL - 6, y + 4);
  }

  let legX = padL;
  seriesList.forEach((s) => {
    ctx.fillStyle = s.color;
    ctx.fillRect(legX, 8, 12, 12);
    ctx.fillStyle = "#0f172a";
    ctx.textAlign = "left";
    ctx.font = "600 11px sans-serif";
    ctx.fillText(s.label, legX + 16, 18);
    legX += ctx.measureText(s.label).width + 28;
  });

  const nCat = Math.max(1, categories.length);
  const nSeries = Math.max(1, seriesList.length);
  const groupW = plotW / nCat;
  const barW = Math.max(8, (groupW * 0.65) / nSeries);

  categories.forEach((cat, cIdx) => {
    const groupCenter = padL + cIdx * groupW + groupW / 2;
    const startX = groupCenter - (nSeries * barW) / 2;

    seriesList.forEach((s, sIdx) => {
      const val = s.values[cIdx] || 0;
      const barH = (Math.max(0, val) / yMax) * plotH;
      const x = startX + sIdx * barW;
      const y = padT + plotH - barH;

      ctx.fillStyle = s.color;
      ctx.fillRect(x, y, barW - 2, barH);
    });

    ctx.fillStyle = "#0f172a";
    ctx.font = "600 11px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(cat, groupCenter, padT + plotH + 18);
  });
}


/* ============================================================================= */
/* LONGITUDINAL REHABILITATION TRACKING & RECOVERY TREND MODULE                  */
/* ============================================================================= */

async function initLongitudinalModule() {
  await loadLongitudinalParticipants();
  await loadLongitudinalData();
}

async function loadLongitudinalParticipants() {
  try {
    const res = await fetch("/api/longitudinal/participants").then((r) => r.json());
    const pts = res.participants || [];
    const select = document.getElementById("select-patient-filter");
    if (select) {
      const currentVal = select.value;
      select.innerHTML = '<option value="ALL">All Participants</option>';
      pts.forEach((p) => {
        const opt = document.createElement("option");
        opt.value = p;
        opt.textContent = `Participant: ${p}`;
        select.appendChild(opt);
      });
      if (pts.includes(currentVal)) {
        select.value = currentVal;
      } else if (pts.length > 0) {
        select.value = pts[0];
      }
    }
  } catch (err) {
    console.error("Failed to load longitudinal participants:", err);
  }
}

async function handleSaveCurrentSession() {
  const data = window._lastAnalysisData;
  const feedback = document.getElementById("save-session-feedback");
  if (!data || !data.model_predictions) {
    if (feedback) {
      feedback.textContent = "Please run an analysis first.";
      feedback.style.color = "#b91c1c";
    }
    return;
  }

  const patientIdInput = document.getElementById("input-session-patient-id");
  const notesInput = document.getElementById("input-session-notes");
  const participantId = (patientIdInput?.value || "ANONYMOUS").trim().toUpperCase();
  const notes = (notesInput?.value || "").trim();

  const preds = data.model_predictions;
  const primaryModel = preds["TCN"] ? "TCN" : Object.keys(preds)[0];
  const mData = preds[primaryModel] || {};

  const t1 = mData.task1_elbow_flexion || {};
  const t2 = mData.task2_forearm_prosup || {};
  const t4 = mData.task4_quality_classification || {};

  const payload = {
    participant_id: participantId,
    exercise_name: data.sample_metadata?.exercise_name || "Elbow Flexion-Extension",
    elbow_rom_deg: t1.full_trial_rom_deg || 0.0,
    forearm_rom_deg: t2.full_trial_rom_deg !== undefined ? t2.full_trial_rom_deg : null,
    max_elbow_angle_deg: t1.max_deg || 0.0,
    min_elbow_angle_deg: t1.min_deg || 0.0,
    max_forearm_angle_deg: t2.max_deg !== undefined ? t2.max_deg : null,
    min_forearm_angle_deg: t2.min_deg !== undefined ? t2.min_deg : null,
    repetition_count: t1.num_repetitions_detected || 0,
    movement_smoothness_sparc: t1.movement_smoothness_sparc !== undefined ? t1.movement_smoothness_sparc : null,
    movement_quality_class: t4.predicted_quality || null,
    movement_quality_confidence_pct: t4.confidence_pct || null,
    model_name: primaryModel,
    model_version: "v2.1-harmonized",
    calibration_status: "Dual IMU 100 Hz Standardized",
    data_quality_status: `Valid (${data.num_timesteps || 1000} frames @ 100 Hz)`,
    sample_source_file: data.filename || null,
    notes: notes,
  };

  try {
    const res = await fetch("/api/longitudinal/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }).then((r) => r.json());

    if (feedback) {
      feedback.textContent = `✓ Logged session (${res.session_id})!`;
      feedback.style.color = "#15803d";
      setTimeout(() => {
        feedback.textContent = "";
        const saveBar = document.getElementById("save-session-bar");
        if (saveBar) saveBar.classList.add("hidden");
      }, 3500);
    }

    await loadLongitudinalParticipants();
    const select = document.getElementById("select-patient-filter");
    if (select) select.value = participantId;
    await loadLongitudinalData();
  } catch (err) {
    console.error("Failed to save session:", err);
    if (feedback) {
      feedback.textContent = "Error saving session: " + err.message;
      feedback.style.color = "#b91c1c";
    }
  }
}

async function loadLongitudinalData() {
  const patientSelect = document.getElementById("select-patient-filter");
  const sDateInput = document.getElementById("input-date-start");
  const eDateInput = document.getElementById("input-date-end");
  const targetRomInput = document.getElementById("input-target-rom");

  const participantId = patientSelect?.value || "ALL";
  const startDate = sDateInput?.value || "";
  const endDate = eDateInput?.value || "";
  const targetRom = parseFloat(targetRomInput?.value || "130.0") || 130.0;

  try {
    const sessUrl = `/api/longitudinal/sessions?participant_id=${encodeURIComponent(participantId)}&start_date=${encodeURIComponent(startDate)}&end_date=${encodeURIComponent(endDate)}`;
    const sessData = await fetch(sessUrl).then((r) => r.json());
    const sessions = sessData.sessions || [];

    let summary = null;
    if (participantId !== "ALL") {
      const sumUrl = `/api/longitudinal/summary/${encodeURIComponent(participantId)}?target_rom_deg=${targetRom}`;
      const sumData = await fetch(sumUrl).then((r) => r.json());
      summary = sumData.summary;
    }

    renderLongitudinalKPIs(summary, sessions, targetRom);
    renderLongitudinalCharts(sessions, summary, targetRom);
    renderLongitudinalTable(sessions);
  } catch (err) {
    console.error("Failed to load longitudinal data:", err);
  }
}

function renderLongitudinalKPIs(summary, sessions, targetRom) {
  const kpiGrid = document.getElementById("longitudinal-kpi-grid");
  if (!kpiGrid) return;

  const count = sessions.length;
  if (count === 0) {
    kpiGrid.innerHTML = `
      <div class="kpi-card">
        <span class="kpi-title">Recorded Sessions</span>
        <span class="kpi-val">0</span>
        <span class="kpi-sub">No sessions found for criteria</span>
      </div>
    `;
    return;
  }

  const baselineRom = summary ? summary.baseline_elbow_rom_deg : sessions[0].elbow_rom_deg;
  const latestRom = summary ? summary.latest_elbow_rom_deg : sessions[sessions.length - 1].elbow_rom_deg;
  const deltaRom = (latestRom - baselineRom).toFixed(1);
  const deltaSign = deltaRom >= 0 ? "+" : "";

  const progressPct = summary ? summary.target_progress_pct : Math.min(100, (latestRom / targetRom) * 100).toFixed(1);

  let trendVal = "Insufficient Data";
  let trendSub = "Requires >= 2 sessions";
  if (summary && summary.linear_trend) {
    const t = summary.linear_trend;
    trendVal = `${t.slope_deg_per_session >= 0 ? '+' : ''}${t.slope_deg_per_session}° / sess`;
    trendSub = `R² = ${t.r_squared} • 95% CI: [${t.slope_ci95_range[0]}°, ${t.slope_ci95_range[1]}°]`;
  }

  const latestSparc = summary && summary.latest_sparc_smoothness !== null
    ? summary.latest_sparc_smoothness
    : (sessions[sessions.length - 1].movement_smoothness_sparc !== null ? sessions[sessions.length - 1].movement_smoothness_sparc : "N/A");

  const correctRate = summary && summary.correct_quality_rate_pct !== null
    ? `${summary.correct_quality_rate_pct}%`
    : "Evaluated";

  kpiGrid.innerHTML = `
    <div class="kpi-card hero-primary">
      <span class="kpi-title">Observed Elbow ROM</span>
      <span class="kpi-val">${latestRom}°</span>
      <span class="kpi-sub">Baseline: ${baselineRom}° (${deltaSign}${deltaRom}° change)</span>
    </div>
    <div class="kpi-card">
      <span class="kpi-title">Functional Goal Progress</span>
      <span class="kpi-val">${progressPct}%</span>
      <span class="kpi-sub">Clinician Target: ${targetRom}°</span>
    </div>
    <div class="kpi-card">
      <span class="kpi-title">Descriptive Trend Rate</span>
      <span class="kpi-val">${trendVal}</span>
      <span class="kpi-sub">${trendSub}</span>
    </div>
    <div class="kpi-card">
      <span class="kpi-title">Total Serial Sessions</span>
      <span class="kpi-val">${count}</span>
      <span class="kpi-sub">${sessions[0].timestamp.split('T')[0]} to ${sessions[sessions.length - 1].timestamp.split('T')[0]}</span>
    </div>
    <div class="kpi-card">
      <span class="kpi-title">Smoothness (SPARC)</span>
      <span class="kpi-val">${latestSparc}</span>
      <span class="kpi-sub">Most recent session score</span>
    </div>
    <div class="kpi-card">
      <span class="kpi-title">Quality Compliance</span>
      <span class="kpi-val">${correctRate}</span>
      <span class="kpi-sub">Correct execution rate</span>
    </div>
  `;
}

function renderLongitudinalCharts(sessions, summary, targetRom) {
  const cRom = document.getElementById("canvas-longitudinal-rom");
  const cEnv = document.getElementById("canvas-longitudinal-envelope");
  const cPro = document.getElementById("canvas-longitudinal-prosup");
  const cSm = document.getElementById("canvas-longitudinal-smoothness");
  const cQual = document.getElementById("canvas-longitudinal-quality");

  if (!sessions.length) return;

  // Session labels (indices or short dates)
  const xLabels = sessions.map((s, idx) => {
    const d = s.timestamp.split("T")[0].slice(5); // "MM-DD"
    return `S${idx + 1} (${d})`;
  });
  const xIndices = sessions.map((_, idx) => idx + 1);

  // 1. Elbow ROM Progression vs Target + Trend
  if (cRom) {
    const romValues = sessions.map((s) => s.elbow_rom_deg);
    const targetValues = sessions.map(() => targetRom);
    const series = [
      { label: "Observed Elbow ROM (°)", values: romValues, color: "#1d4ed8", lineWidth: 2.8 },
      { label: `Target Goal (${targetRom}°)`, values: targetValues, color: "#dc2626", lineWidth: 1.8 },
    ];

    if (summary && summary.linear_trend && sessions.length >= 2) {
      const m = summary.linear_trend.slope_deg_per_session;
      const b = summary.linear_trend.intercept_deg;
      const trendValues = xIndices.map((x) => Number((m * x + b).toFixed(1)));
      series.push({
        label: `Descriptive Trend (Slope: ${m >= 0 ? '+' : ''}${m}°/sess, R²=${summary.linear_trend.r_squared})`,
        values: trendValues,
        color: "#d97706",
        lineWidth: 2.0,
      });
    }

    drawMultiLineChart(cRom, xIndices, series, "Rehabilitation Session #", "Elbow Range of Motion (°)");
  }

  // 2. Elbow Angle Envelope (Max vs Min)
  if (cEnv) {
    const maxAngles = sessions.map((s) => s.max_elbow_angle_deg);
    const minAngles = sessions.map((s) => s.min_elbow_angle_deg);
    drawMultiLineChart(
      cEnv,
      xIndices,
      [
        { label: "Peak Flexion (°)", values: maxAngles, color: "#059669", lineWidth: 2.2 },
        { label: "Extension Min (°)", values: minAngles, color: "#e11d48", lineWidth: 2.2 },
      ],
      "Session #",
      "Joint Angle Envelope (°)"
    );
  }

  // 3. Forearm Pronation/Supination ROM
  if (cPro) {
    const prosupValues = sessions.map((s) => (s.forearm_rom_deg !== null ? s.forearm_rom_deg : null));
    // Filter out sessions without forearm measurement to avoid fabricating 0
    const validPairs = sessions
      .map((s, idx) => ({ x: idx + 1, val: s.forearm_rom_deg }))
      .filter((p) => p.val !== null);

    if (validPairs.length > 0) {
      drawMultiLineChart(
        cPro,
        validPairs.map((p) => p.x),
        [{ label: "Forearm Rotational ROM (°)", values: validPairs.map((p) => p.val), color: "#7c3aed", lineWidth: 2.5 }],
        "Session #",
        "Forearm ROM (°)"
      );
    } else {
      const ctx = cPro.getContext("2d");
      ctx.clearRect(0, 0, cPro.width, cPro.height);
      ctx.fillStyle = "#64748b";
      ctx.font = "12px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Forearm rotation not evaluated in selected sessions", cPro.width / 2, cPro.height / 2);
    }
  }

  // 4. Repetition Count & Smoothness (SPARC)
  if (cSm) {
    const repCounts = sessions.map((s) => s.repetition_count);
    const sparcVals = sessions.map((s) => (s.movement_smoothness_sparc !== null ? s.movement_smoothness_sparc : -4.0));
    drawMultiLineChart(
      cSm,
      xIndices,
      [
        { label: "Smoothness (SPARC)", values: sparcVals, color: "#0891b2", lineWidth: 2.2 },
        { label: "Repetitions Count", values: repCounts, color: "#ea580c", lineWidth: 2.0 },
      ],
      "Session #",
      "Reps & Smoothness"
    );
  }

  // 5. Movement Quality Execution
  if (cQual) {
    const qualVals = sessions.map((s) => {
      if (s.movement_quality_class === "Correct Execution") {
        return s.movement_quality_confidence_pct || 100;
      }
      return 0;
    });
    drawMultiLineChart(
      cQual,
      xIndices,
      [{ label: "Movement Quality Score (% Correct)", values: qualVals, color: "#10b981", lineWidth: 2.4 }],
      "Session #",
      "Execution Quality (%)"
    );
  }
}

function renderLongitudinalTable(sessions) {
  const table = document.getElementById("table-sessions-history");
  const countBadge = document.getElementById("session-records-count");
  if (!table) return;

  if (countBadge) countBadge.textContent = sessions.length;
  const tbody = table.querySelector("tbody");
  if (!tbody) return;

  tbody.innerHTML = "";
  if (!sessions.length) {
    tbody.innerHTML = '<tr><td colspan="11" style="text-align:center; color:#64748b; padding:1.5rem;">No session records found matching filter criteria.</td></tr>';
    return;
  }

  // Render in reverse chronological order (newest first)
  const reversed = [...sessions].reverse();
  reversed.forEach((s) => {
    const tr = document.createElement("tr");
    const dStr = s.timestamp.replace("T", " ").replace("Z", "");
    const proRom = s.forearm_rom_deg !== null && s.forearm_rom_deg !== undefined ? `${s.forearm_rom_deg}°` : '<span style="color:#94a3b8;">N/A</span>';
    const sparc = s.movement_smoothness_sparc !== null && s.movement_smoothness_sparc !== undefined ? s.movement_smoothness_sparc.toFixed(2) : '<span style="color:#94a3b8;">N/A</span>';
    const isCorrect = s.movement_quality_class === "Correct Execution";
    const qualBadge = s.movement_quality_class
      ? `<span class="badge" style="background:${isCorrect ? '#dcfce7' : '#fee2e2'}; color:${isCorrect ? '#15803d' : '#b91c1c'}; font-size:0.75rem;">${s.movement_quality_class}</span>`
      : '<span style="color:#94a3b8;">Not Assessed</span>';

    tr.innerHTML = `
      <td style="font-family:monospace; font-size:0.78rem;">${dStr}</td>
      <td style="font-weight:700; color:#1e40af;">${s.participant_id}</td>
      <td>${s.exercise_name}</td>
      <td style="font-weight:700; color:#15803d;">${s.elbow_rom_deg}°</td>
      <td>${proRom}</td>
      <td style="font-size:0.78rem;">${s.min_elbow_angle_deg}° – ${s.max_elbow_angle_deg}°</td>
      <td>${s.repetition_count}</td>
      <td>${sparc}</td>
      <td>${qualBadge}</td>
      <td style="font-size:0.75rem; color:#64748b;">${s.model_name} (${s.model_version})</td>
      <td>
        <button type="button" class="btn-secondary" style="padding:0.2rem 0.5rem; font-size:0.72rem; color:#b91c1c; border-color:#fca5a5;" onclick="handleDeleteSession('${s.session_id}')">
          Delete
        </button>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

async function handleDeleteSession(sessionId) {
  if (!confirm(`Delete rehabilitation session record ${sessionId}?`)) return;
  try {
    const res = await fetch(`/api/longitudinal/sessions/${encodeURIComponent(sessionId)}`, { method: "DELETE" }).then((r) => r.json());
    await loadLongitudinalParticipants();
    await loadLongitudinalData();
  } catch (err) {
    alert("Failed to delete session: " + err.message);
  }
}


/* ============================================================================= */
/* VERIFIED ESP32 DUAL-IMU HARDWARE TRANSPORT & LIVE STREAMING MODULE            */
/* ============================================================================= */

const ESP32_STATE = {
  DISCONNECTED: "Disconnected",
  SEARCHING: "Searching",
  AWAITING: "Awaiting Response",
  CONNECTED_VERIFIED: "Connected — Verified",
  FAILED: "Connection Failed",
  DEVICE_DISCONNECTED: "Device Disconnected",
  INVALID_DATA: "Invalid Sensor Data",
  SIMULATION: "Simulation Replay (Offline)",
};

let esp32State = ESP32_STATE.DISCONNECTED;
let activeSerialPort = null;
let activeSerialReader = null;
let activeBleDevice = null;
let activeBleCharacteristic = null;
let simIntervalTimer = null;
let keepaliveTimer = null;

let isRecordingEspStream = false;
let recordedPacketRows = [];

let livePacketCount = 0;
let liveDroppedCount = 0;
let lastSeqNumber = null;
let lastPacketTimestamp = null;
let recentTimestamps = [];
let liveMeasuredRateHz = 0.0;

// Scope ring buffer: 100 points
const SCOPE_BUFFER_LEN = 120;
let scopeBuffer = {
  t: [],
  rua_ax: [], rua_ay: [], rua_az: [],
  rla_ax: [], rla_ay: [], rla_az: [],
};

function updateEsp32UIState(state, message = "") {
  esp32State = state;
  const indicator = document.getElementById("esp32-conn-indicator");
  const label = document.getElementById("esp32-status-label");
  const badge = document.getElementById("esp32-transport-badge");
  const btnSerial = document.getElementById("btn-esp32-connect-serial");
  const btnBle = document.getElementById("btn-esp32-connect-ble");
  const btnDisconnect = document.getElementById("btn-esp32-disconnect");
  const btnSim = document.getElementById("btn-esp32-sim-replay");
  const btnRecord = document.getElementById("btn-esp32-record");
  const btnSave = document.getElementById("btn-esp32-save-csv");
  const warningBox = document.getElementById("esp32-warning-banner");

  if (!indicator || !label) return;

  // Clean indicator classes
  indicator.className = "status-indicator";

  switch (state) {
    case ESP32_STATE.DISCONNECTED:
      indicator.classList.add("status-offline");
      label.textContent = "Disconnected (Standby)";
      label.style.color = "#475569";
      if (badge) {
        badge.textContent = "Physical Transport Ready";
        badge.style.background = "#f1f5f9";
        badge.style.color = "#475569";
      }
      if (btnSerial) btnSerial.disabled = false;
      if (btnBle) btnBle.disabled = false;
      if (btnDisconnect) btnDisconnect.disabled = true;
      if (btnRecord) btnRecord.disabled = true;
      if (btnSave) btnSave.disabled = recordedPacketRows.length === 0;
      if (warningBox) warningBox.classList.add("hidden");
      break;

    case ESP32_STATE.SEARCHING:
      indicator.classList.add("status-searching");
      label.textContent = message || "Searching for physical ESP32 device...";
      label.style.color = "#d97706";
      if (btnSerial) btnSerial.disabled = true;
      if (btnBle) btnBle.disabled = true;
      if (btnDisconnect) btnDisconnect.disabled = false;
      break;

    case ESP32_STATE.AWAITING:
      indicator.classList.add("status-awaiting");
      label.textContent = message || "Port opened. Awaiting verified packet handshake...";
      label.style.color = "#2563eb";
      if (btnSerial) btnSerial.disabled = true;
      if (btnBle) btnBle.disabled = true;
      if (btnDisconnect) btnDisconnect.disabled = false;
      break;

    case ESP32_STATE.CONNECTED_VERIFIED:
      indicator.classList.add("status-online");
      label.textContent = "Connected — Verified (ESP32 Dual-IMU Active @ 100 Hz)";
      label.style.color = "#15803d";
      if (badge) {
        badge.textContent = "✓ Verified Hardware Connection";
        badge.style.background = "#dcfce7";
        badge.style.color = "#15803d";
      }
      if (btnSerial) btnSerial.disabled = true;
      if (btnBle) btnBle.disabled = true;
      if (btnDisconnect) btnDisconnect.disabled = false;
      if (btnRecord) btnRecord.disabled = false;
      break;

    case ESP32_STATE.FAILED:
      indicator.classList.add("status-failed");
      label.textContent = `Connection Failed: ${message || 'No response from device'}`;
      label.style.color = "#b91c1c";
      if (badge) {
        badge.textContent = "⚠ Connection Failed";
        badge.style.background = "#fee2e2";
        badge.style.color = "#b91c1c";
      }
      if (btnSerial) btnSerial.disabled = false;
      if (btnBle) btnBle.disabled = false;
      if (btnDisconnect) btnDisconnect.disabled = true;
      if (btnRecord) btnRecord.disabled = true;
      break;

    case ESP32_STATE.DEVICE_DISCONNECTED:
      indicator.classList.add("status-offline");
      label.textContent = `Device Disconnected: ${message || 'Physical transport closed'}`;
      label.style.color = "#991b1b";
      if (btnSerial) btnSerial.disabled = false;
      if (btnBle) btnBle.disabled = false;
      if (btnDisconnect) btnDisconnect.disabled = true;
      if (btnRecord) btnRecord.disabled = true;
      break;

    case ESP32_STATE.INVALID_DATA:
      indicator.classList.add("status-failed");
      label.textContent = `Invalid Sensor Data: ${message}`;
      label.style.color = "#ea580c";
      break;

    case ESP32_STATE.SIMULATION:
      indicator.classList.add("status-sim");
      label.textContent = "Simulation Mode — Offline Benchmark Replay (Not Real Hardware)";
      label.style.color = "#7c3aed";
      if (badge) {
        badge.textContent = "Offline Benchmark Simulation";
        badge.style.background = "#f3e8ff";
        badge.style.color = "#7c3aed";
      }
      if (btnSerial) btnSerial.disabled = true;
      if (btnBle) btnBle.disabled = true;
      if (btnDisconnect) btnDisconnect.disabled = false;
      if (btnRecord) btnRecord.disabled = false;
      break;
  }
}

async function connectEsp32Serial() {
  if (!("serial" in navigator)) {
    updateEsp32UIState(
      ESP32_STATE.FAILED,
      "Web Serial API is not supported in this browser. Please use Chrome or Edge on desktop."
    );
    return;
  }

  updateEsp32UIState(ESP32_STATE.SEARCHING, "Requesting USB Serial COM port from user...");

  try {
    const port = await navigator.serial.requestPort();
    activeSerialPort = port;

    // Listen for transport disconnect
    port.addEventListener("disconnect", () => {
      handleTransportClosed("USB cable unplugged or serial port closed.");
    });

    updateEsp32UIState(ESP32_STATE.AWAITING, "Opening COM port @ 115200 baud & awaiting handshake...");
    await port.open({ baudRate: 115200 });

    // Send handshake PING
    try {
      const textEncoder = new TextEncoderStream();
      const writableStreamClosed = textEncoder.readable.pipeTo(port.writable);
      const writer = textEncoder.writable.getWriter();
      await writer.write("PING\n");
      writer.releaseLock();
    } catch (wErr) {
      console.warn("Handshake write error:", wErr);
    }

    // Set connection timeout (3.5 seconds)
    let firstPacketReceived = false;
    const handshakeTimeout = setTimeout(() => {
      if (!firstPacketReceived && esp32State === ESP32_STATE.AWAITING) {
        updateEsp32UIState(
          ESP32_STATE.FAILED,
          "Handshake timeout (3.5s). Port opened but no valid 64-byte packets received from ESP32."
        );
        disconnectEsp32Device();
      }
    }, 3500);

    // Read byte stream
    const reader = port.readable.getReader();
    activeSerialReader = reader;
    let byteBuffer = new Uint8Array(0);

    startKeepaliveMonitor();

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      if (value) {
        // Append bytes
        const newBuf = new Uint8Array(byteBuffer.length + value.length);
        newBuf.set(byteBuffer, 0);
        newBuf.set(value, byteBuffer.length);
        byteBuffer = newBuf;

        // Search for magic header 0xAA55AA55
        while (byteBuffer.length >= 64) {
          let magicIdx = -1;
          for (let i = 0; i <= byteBuffer.length - 64; i++) {
            if (
              byteBuffer[i] === 0x55 &&
              byteBuffer[i + 1] === 0xaa &&
              byteBuffer[i + 2] === 0x55 &&
              byteBuffer[i + 3] === 0xaa
            ) {
              magicIdx = i;
              break;
            }
          }

          if (magicIdx === -1) {
            // Keep last 3 bytes for boundary search
            byteBuffer = byteBuffer.slice(Math.max(0, byteBuffer.length - 3));
            break;
          }

          if (magicIdx > 0) {
            byteBuffer = byteBuffer.slice(magicIdx);
          }

          if (byteBuffer.length < 64) break;

          const packetSlice = byteBuffer.slice(0, 64);
          byteBuffer = byteBuffer.slice(64);

          const parsed = parseBinarySensorPacket(packetSlice);
          if (parsed && parsed.is_valid) {
            firstPacketReceived = true;
            clearTimeout(handshakeTimeout);
            processIncomingPacket(parsed);
          }
        }
      }
    }
  } catch (err) {
    console.error("Serial connection error:", err);
    if (err.name === "NotFoundError") {
      updateEsp32UIState(ESP32_STATE.DISCONNECTED, "Port selection cancelled by user.");
    } else {
      updateEsp32UIState(ESP32_STATE.FAILED, err.message || "Failed to communicate with serial port.");
    }
  }
}

async function connectEsp32Bluetooth() {
  if (!("bluetooth" in navigator)) {
    updateEsp32UIState(
      ESP32_STATE.FAILED,
      "Web Bluetooth API is not supported in this browser. Please use Chrome or Edge."
    );
    return;
  }

  updateEsp32UIState(ESP32_STATE.SEARCHING, "Requesting ESP32 BLE device from user...");

  try {
    const device = await navigator.bluetooth.requestDevice({
      filters: [{ namePrefix: "ESP32" }, { namePrefix: "REHAB" }],
      optionalServices: ["4fafc201-1fb5-459e-8fcc-c5c9c331914b"],
    });

    activeBleDevice = device;
    device.addEventListener("gattserverdisconnected", () => {
      handleTransportClosed("BLE device disconnected.");
    });

    updateEsp32UIState(ESP32_STATE.AWAITING, "Connecting to GATT server...");
    const server = await device.gatt.connect();
    const service = await server.getPrimaryService("4fafc201-1fb5-459e-8fcc-c5c9c331914b");
    const characteristic = await service.getCharacteristic("beb5483e-36e1-4688-b7f5-ea07361b26a8");
    activeBleCharacteristic = characteristic;

    let firstPacket = false;
    const bleTimeout = setTimeout(() => {
      if (!firstPacket && esp32State === ESP32_STATE.AWAITING) {
        updateEsp32UIState(ESP32_STATE.FAILED, "BLE Handshake timeout: No notifications received.");
        disconnectEsp32Device();
      }
    }, 4000);

    await characteristic.startNotifications();
    startKeepaliveMonitor();

    characteristic.addEventListener("characteristicvaluechanged", (event) => {
      const dataView = event.target.value;
      if (dataView.byteLength === 64) {
        const u8 = new Uint8Array(dataView.buffer);
        const parsed = parseBinarySensorPacket(u8);
        if (parsed && parsed.is_valid) {
          firstPacket = true;
          clearTimeout(bleTimeout);
          processIncomingPacket(parsed);
        }
      }
    });
  } catch (err) {
    console.error("BLE connection error:", err);
    if (err.name === "NotFoundError") {
      updateEsp32UIState(ESP32_STATE.DISCONNECTED, "Device selection cancelled by user.");
    } else {
      updateEsp32UIState(ESP32_STATE.FAILED, err.message || "Failed to connect via BLE.");
    }
  }
}

function parseBinarySensorPacket(bytes) {
  if (bytes.length !== 64) return null;
  const dv = new DataView(bytes.buffer, bytes.byteOffset, 64);

  const magic = dv.getUint32(0, true);
  // Little endian magic: 0xAA55AA55
  if (magic !== 0xaa55aa55) return null;

  const t_ms = dv.getUint32(4, true);
  const seq = dv.getUint32(8, true);

  const floats = [];
  for (let i = 0; i < 12; i++) {
    floats.push(Number(dv.getFloat32(12 + i * 4, true).toFixed(4)));
  }

  const status_flags = dv.getUint16(60, true);
  const checksum = dv.getUint16(62, true);

  // Validate checksum
  let computedSum = 0;
  for (let i = 0; i < 60; i++) {
    computedSum += bytes[i];
  }
  computedSum = computedSum & 0xffff;
  if (checksum !== 0 && checksum !== computedSum) {
    console.warn("Packet checksum mismatch", checksum, computedSum);
    return null;
  }

  const channels = {
    RUA_Ax: floats[0], RUA_Ay: floats[1], RUA_Az: floats[2],
    RUA_Gx: floats[3], RUA_Gy: floats[4], RUA_Gz: floats[5],
    RLA_Ax: floats[6], RLA_Ay: floats[7], RLA_Az: floats[8],
    RLA_Gx: floats[9], RLA_Gy: floats[10], RLA_Gz: floats[11],
  };

  return {
    is_valid: true,
    t_ms: t_ms,
    seq: seq,
    status_flags: status_flags,
    channels: channels,
  };
}

function processIncomingPacket(packet) {
  const now = performance.now();
  lastPacketTimestamp = now;

  // Track sequence drops
  if (lastSeqNumber !== null) {
    const expected = (lastSeqNumber + 1) >>> 0;
    if (packet.seq !== expected && packet.seq > lastSeqNumber) {
      liveDroppedCount += (packet.seq - lastSeqNumber - 1);
    }
  }
  lastSeqNumber = packet.seq;
  livePacketCount++;

  // Update sampling rate
  recentTimestamps.push(now);
  if (recentTimestamps.length > 50) recentTimestamps.shift();
  if (recentTimestamps.length >= 10) {
    const spanMs = recentTimestamps[recentTimestamps.length - 1] - recentTimestamps[0];
    if (spanMs > 0) {
      liveMeasuredRateHz = Number(((recentTimestamps.length - 1) / (spanMs / 1000)).toFixed(1));
    }
  }

  // Dual-IMU Health Checks
  const ch = packet.channels;
  const ruaNorm = Math.sqrt(ch.RUA_Ax ** 2 + ch.RUA_Ay ** 2 + ch.RUA_Az ** 2);
  const rlaNorm = Math.sqrt(ch.RLA_Ax ** 2 + ch.RLA_Ay ** 2 + ch.RLA_Az ** 2);

  const ruaActive = ruaNorm > 0.05 && !(packet.status_flags & 0x01);
  const rlaActive = rlaNorm > 0.05 && !(packet.status_flags & 0x02);

  // Update Status Badges
  const badgeRua = document.getElementById("badge-rua-status");
  const badgeRla = document.getElementById("badge-rla-status");
  const warningBox = document.getElementById("esp32-warning-banner");
  const warningDesc = document.getElementById("esp32-warning-desc");

  if (badgeRua) {
    badgeRua.textContent = ruaActive ? "Active (100 Hz)" : "Silent / Disconnected";
    badgeRua.style.background = ruaActive ? "#dcfce7" : "#fee2e2";
    badgeRua.style.color = ruaActive ? "#15803d" : "#b91c1c";
  }
  if (badgeRla) {
    badgeRla.textContent = rlaActive ? "Active (100 Hz)" : "Silent / Disconnected";
    badgeRla.style.background = rlaActive ? "#dcfce7" : "#fee2e2";
    badgeRla.style.color = rlaActive ? "#15803d" : "#b91c1c";
  }

  if (!ruaActive || !rlaActive) {
    if (warningBox && warningDesc) {
      warningBox.classList.remove("hidden");
      const missing = !rlaActive ? "Forearm IMU (RLA)" : "Upper Arm IMU (RUA)";
      warningDesc.textContent = `Single-IMU detected: ${missing} is inactive or uncalibrated. 12-channel elbow joint kinematics require dual IMUs.`;
    }
  } else {
    if (warningBox) warningBox.classList.add("hidden");
  }

  // Transition to Connected — Verified if not in simulation
  if (esp32State !== ESP32_STATE.SIMULATION && esp32State !== ESP32_STATE.CONNECTED_VERIFIED) {
    updateEsp32UIState(ESP32_STATE.CONNECTED_VERIFIED);
  }

  // Update Diagnostics Text
  const rateEl = document.getElementById("esp32-live-rate");
  const packEl = document.getElementById("esp32-live-packets");
  const dropEl = document.getElementById("esp32-live-drops");
  const telRua = document.getElementById("telemetry-rua");
  const telRla = document.getElementById("telemetry-rla");

  if (rateEl) rateEl.textContent = `${liveMeasuredRateHz} Hz`;
  if (packEl) packEl.textContent = livePacketCount;
  if (dropEl) dropEl.textContent = liveDroppedCount;

  if (telRua) {
    telRua.innerHTML = `Acc: [${ch.RUA_Ax.toFixed(2)}, ${ch.RUA_Ay.toFixed(2)}, ${ch.RUA_Az.toFixed(2)}] m/s²<br>Gyr: [${ch.RUA_Gx.toFixed(2)}, ${ch.RUA_Gy.toFixed(2)}, ${ch.RUA_Gz.toFixed(2)}] rad/s`;
  }
  if (telRla) {
    telRla.innerHTML = `Acc: [${ch.RLA_Ax.toFixed(2)}, ${ch.RLA_Ay.toFixed(2)}, ${ch.RLA_Az.toFixed(2)}] m/s²<br>Gyr: [${ch.RLA_Gx.toFixed(2)}, ${ch.RLA_Gy.toFixed(2)}, ${ch.RLA_Gz.toFixed(2)}] rad/s`;
  }

  // Record if active
  if (isRecordingEspStream) {
    recordedPacketRows.push([
      packet.t_ms, packet.seq,
      ch.RUA_Ax, ch.RUA_Ay, ch.RUA_Az,
      ch.RUA_Gx, ch.RUA_Gy, ch.RUA_Gz,
      ch.RLA_Ax, ch.RLA_Ay, ch.RLA_Az,
      ch.RLA_Gx, ch.RLA_Gy, ch.RLA_Gz,
    ]);
    if (livePacketCount % 10 === 0) {
      const btn = document.getElementById("btn-esp32-record");
      if (btn) btn.textContent = `⏹ Stop Recording (${recordedPacketRows.length} frames)`;
    }
  }

  // Append to Scope Buffer
  scopeBuffer.t.push(livePacketCount);
  scopeBuffer.rua_ax.push(ch.RUA_Ax);
  scopeBuffer.rua_ay.push(ch.RUA_Ay);
  scopeBuffer.rua_az.push(ch.RUA_Az);
  scopeBuffer.rla_ax.push(ch.RLA_Ax);
  scopeBuffer.rla_ay.push(ch.RLA_Ay);
  scopeBuffer.rla_az.push(ch.RLA_Az);

  if (scopeBuffer.t.length > SCOPE_BUFFER_LEN) {
    scopeBuffer.t.shift();
    scopeBuffer.rua_ax.shift();
    scopeBuffer.rua_ay.shift();
    scopeBuffer.rua_az.shift();
    scopeBuffer.rla_ax.shift();
    scopeBuffer.rla_ay.shift();
    scopeBuffer.rla_az.shift();
  }

  // Draw Live Scope at throttled 30 fps
  if (livePacketCount % 3 === 0) {
    drawLiveScope();
  }
}

function drawLiveScope() {
  const canvas = document.getElementById("canvas-esp32-scope");
  if (!canvas || scopeBuffer.t.length < 2) return;

  const series = [
    { label: "RUA Acc X", values: scopeBuffer.rua_ax, color: "#ef4444", lineWidth: 1.5 },
    { label: "RUA Acc Y", values: scopeBuffer.rua_ay, color: "#10b981", lineWidth: 1.5 },
    { label: "RUA Acc Z", values: scopeBuffer.rua_az, color: "#3b82f6", lineWidth: 1.5 },
    { label: "RLA Acc X", values: scopeBuffer.rla_ax, color: "#f97316", lineWidth: 1.5 },
    { label: "RLA Acc Y", values: scopeBuffer.rla_ay, color: "#8b5cf6", lineWidth: 1.5 },
    { label: "RLA Acc Z", values: scopeBuffer.rla_az, color: "#06b6d4", lineWidth: 1.5 },
  ];

  drawMultiLineChart(canvas, scopeBuffer.t, series, "Frame #", "Acceleration (m/s²)");
}

function startKeepaliveMonitor() {
  if (keepaliveTimer) clearInterval(keepaliveTimer);
  keepaliveTimer = setInterval(() => {
    if (esp32State === ESP32_STATE.CONNECTED_VERIFIED) {
      if (lastPacketTimestamp && performance.now() - lastPacketTimestamp > 1200) {
        updateEsp32UIState(
          ESP32_STATE.DEVICE_DISCONNECTED,
          "Stream stalled: no packets received for >1.2 seconds."
        );
      }
    }
  }, 500);
}

function handleTransportClosed(reason) {
  if (keepaliveTimer) clearInterval(keepaliveTimer);
  updateEsp32UIState(ESP32_STATE.DEVICE_DISCONNECTED, reason);
  activeSerialReader = null;
  activeSerialPort = null;
  activeBleCharacteristic = null;
  activeBleDevice = null;
}

async function disconnectEsp32Device() {
  if (simIntervalTimer) {
    clearInterval(simIntervalTimer);
    simIntervalTimer = null;
  }
  if (keepaliveTimer) {
    clearInterval(keepaliveTimer);
    keepaliveTimer = null;
  }

  if (activeSerialReader) {
    try { await activeSerialReader.cancel(); } catch (e) {}
    activeSerialReader = null;
  }
  if (activeSerialPort) {
    try { await activeSerialPort.close(); } catch (e) {}
    activeSerialPort = null;
  }
  if (activeBleDevice && activeBleDevice.gatt.connected) {
    try { activeBleDevice.gatt.disconnect(); } catch (e) {}
    activeBleDevice = null;
    activeBleCharacteristic = null;
  }

  updateEsp32UIState(ESP32_STATE.DISCONNECTED);
}

function toggleSimulationReplay() {
  if (esp32State === ESP32_STATE.SIMULATION) {
    disconnectEsp32Device();
    return;
  }

  disconnectEsp32Device();
  updateEsp32UIState(ESP32_STATE.SIMULATION);

  // Generate synthetic cyclical 100 Hz dual-IMU test stream
  let simFrame = 0;
  simIntervalTimer = setInterval(() => {
    simFrame++;
    const t = simFrame * 0.01; // 100 Hz
    const ruaAx = Number((9.81 * Math.sin(t * 1.5)).toFixed(3));
    const ruaAy = Number((9.81 * Math.cos(t * 1.5)).toFixed(3));
    const ruaAz = Number((0.5 * Math.sin(t * 3.0)).toFixed(3));
    const ruaGx = Number((1.2 * Math.cos(t * 1.5)).toFixed(3));
    const ruaGy = Number((0.2 * Math.sin(t * 1.5)).toFixed(3));
    const ruaGz = Number((0.1 * Math.cos(t * 1.5)).toFixed(3));

    const rlaAx = Number((9.81 * Math.sin(t * 1.5 + 0.3)).toFixed(3));
    const rlaAy = Number((9.81 * Math.cos(t * 1.5 + 0.3)).toFixed(3));
    const rlaAz = Number((0.6 * Math.cos(t * 3.0)).toFixed(3));
    const rlaGx = Number((1.5 * Math.cos(t * 1.5 + 0.3)).toFixed(3));
    const rlaGy = Number((0.3 * Math.sin(t * 1.5 + 0.3)).toFixed(3));
    const rlaGz = Number((0.15 * Math.cos(t * 1.5 + 0.3)).toFixed(3));

    const simPacket = {
      is_valid: true,
      t_ms: simFrame * 10,
      seq: simFrame,
      status_flags: 0,
      channels: {
        RUA_Ax: ruaAx, RUA_Ay: ruaAy, RUA_Az: ruaAz,
        RUA_Gx: ruaGx, RUA_Gy: ruaGy, RUA_Gz: ruaGz,
        RLA_Ax: rlaAx, RLA_Ay: rlaAy, RLA_Az: rlaAz,
        RLA_Gx: rlaGx, RLA_Gy: rlaGy, RLA_Gz: rlaGz,
      },
    };

    processIncomingPacket(simPacket);
  }, 10);
}

function toggleEsp32Recording() {
  const btn = document.getElementById("btn-esp32-record");
  const btnSave = document.getElementById("btn-esp32-save-csv");
  const btnAnalyze = document.getElementById("btn-esp32-analyze-stream");
  isRecordingEspStream = !isRecordingEspStream;

  if (isRecordingEspStream) {
    recordedPacketRows = [];
    if (btn) {
      btn.classList.add("btn-recording-active");
      btn.classList.remove("btn-record");
      btn.textContent = "⏹ Stop Recording (0 frames)";
    }
    if (btnSave) btnSave.disabled = true;
    if (btnAnalyze) btnAnalyze.disabled = true;
  } else {
    if (btn) {
      btn.classList.remove("btn-recording-active");
      btn.classList.add("btn-record");
      btn.textContent = "⏺ Start Recording Stream";
    }
    if (btnSave) {
      btnSave.disabled = false;
      btnSave.textContent = `Save Recorded CSV (${recordedPacketRows.length} frames)`;
    }
    if (btnAnalyze) {
      btnAnalyze.disabled = recordedPacketRows.length < 50;
      btnAnalyze.textContent = `Analyze Recorded Stream (${recordedPacketRows.length} frames)`;
    }
  }
}

function saveEsp32RecordedCsv() {
  if (!recordedPacketRows.length) {
    alert("No stream frames recorded yet. Start recording first.");
    return;
  }

  let csvContent = "t_ms,seq,RUA_Ax,RUA_Ay,RUA_Az,RUA_Gx,RUA_Gy,RUA_Gz,RLA_Ax,RLA_Ay,RLA_Az,RLA_Gx,RLA_Gy,RLA_Gz\n";
  recordedPacketRows.forEach((row) => {
    csvContent += row.join(",") + "\n";
  });

  const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `esp32_dual_imu_stream_${Date.now()}.csv`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}


/* ============================================================================= */

async function analyzeRecordedLiveStream() {
  if (!recordedPacketRows || recordedPacketRows.length === 0) {
    alert("No stream frames recorded yet. Please click 'Start Recording Stream', perform the movement, click 'Stop Recording', and then click 'Analyze Recorded Stream'.");
    return;
  }

  const modelSelect = document.getElementById("model-select");
  const selectedModel = modelSelect ? modelSelect.value : "COMPARE_ALL";
  const spinner = document.getElementById("analysis-spinner");

  if (spinner) spinner.classList.remove("hidden");
  updateStatus(`Analyzing ${recordedPacketRows.length} live dual-IMU frames across models...`);

  try {
    let csvContent = "t_ms,seq,RUA_Ax,RUA_Ay,RUA_Az,RUA_Gx,RUA_Gy,RUA_Gz,RLA_Ax,RLA_Ay,RLA_Az,RLA_Gx,RLA_Gy,RLA_Gz\n";
    recordedPacketRows.forEach((row) => {
      csvContent += row.join(",") + "\n";
    });

    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const formData = new FormData();
    formData.append("file", blob, `esp32_live_recording_${Date.now()}.csv`);
    formData.append("selected_model", selectedModel);

    const r = await fetch("/api/analyze/upload", {
      method: "POST",
      body: formData,
    });

    if (!r.ok) {
      const errText = await r.text();
      throw new Error(errText);
    }

    const res = await r.json();
    renderAnalysisResults(res);
    updateStatus(`Analysis complete for live recording (${recordedPacketRows.length} frames).`);

    // Smoothly scroll down to Section 3: Joint Motion Estimation
    const sec3 = document.getElementById("sec-3-joint-motion");
    if (sec3) {
      sec3.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  } catch (err) {
    console.error("Live stream analysis error:", err);
    updateStatus(`Live stream analysis error: ${err.message}`, true);
    alert(`Analysis error: ${err.message}`);
  } finally {
    if (spinner) spinner.classList.add("hidden");
  }
}

/* 8. FINAL ASSESSMENT & CLINICAL FEEDBACK MODULE                                */
/* ============================================================================= */

function renderFinalAssessment(data) {
  const container = document.getElementById("final-assessment-container");
  if (!container) return;

  const fa = data.final_clinical_assessment;
  if (!fa) {
    container.innerHTML = '<div style="background:#f8fafc; border:1px solid var(--border); border-radius:8px; padding:1.25rem; text-align:center; color:#64748b;">No clinical assessment generated.</div>';
    return;
  }

  const recList = (fa.actionable_recommendations || []).map((r) => `<li style="margin-bottom:0.4rem;">${r}</li>`).join("");

  const compBadgeColor = fa.compensatory_badge === "Compliant" ? "#15803d" : "#b91c1c";
  const compBadgeBg = fa.compensatory_badge === "Compliant" ? "#dcfce7" : "#fee2e2";

  container.innerHTML = `
    <!-- Summary Top Grid -->
    <div class="kpi-grid" style="margin-bottom:1rem;">
      <div class="kpi-card hero-primary">
        <span class="kpi-title">Functional ROM Tier</span>
        <span class="kpi-val" style="font-size:1.15rem; color:#1e40af;">${fa.rom_badge}</span>
        <span class="kpi-sub">${fa.rom_status}</span>
      </div>
      <div class="kpi-card">
        <span class="kpi-title">Milestone Goal (130°)</span>
        <span class="kpi-val">${fa.rom_achievement_pct}%</span>
        <span class="kpi-sub">${fa.measured_elbow_rom_deg}° / 130.0° achieved</span>
      </div>
      <div class="kpi-card">
        <span class="kpi-title">Motor Smoothness (SPARC)</span>
        <span class="kpi-val">${fa.smoothness_badge}</span>
        <span class="kpi-sub">${fa.smoothness_status} (Score: ${fa.smoothness_sparc !== null ? fa.smoothness_sparc : 'N/A'})</span>
      </div>
      <div class="kpi-card">
        <span class="kpi-title">Compensatory Substitution Risk</span>
        <span class="kpi-val" style="color:${compBadgeColor};">${fa.compensatory_badge}</span>
        <span class="kpi-sub">${fa.compensatory_risk}</span>
      </div>
    </div>

    <!-- Recommendations Box -->
    <div style="background:#ffffff; border:1px solid var(--border); border-radius:8px; padding:1.15rem; margin-bottom:1rem;">
      <h4 style="margin:0 0 0.6rem 0; color:#1e3a8a; font-size:1.05rem;">Evidence-Based Therapeutic Recommendations</h4>
      <ul style="margin:0; padding-left:1.25rem; color:#334155; font-size:0.88rem; line-height:1.6;">
        ${recList}
      </ul>
    </div>

    <!-- Provenance & Disclaimer Box -->
    <div style="background:#f8fafc; border:1px solid var(--border); border-radius:8px; padding:0.85rem 1rem; font-size:0.8rem; color:#64748b;">
      <div style="display:flex; justify-content:space-between; flex-wrap:wrap; gap:0.5rem; margin-bottom:0.4rem;">
        <span><strong>Evaluating Model:</strong> ${fa.evaluating_model}</span>
        <span><strong>Exercise:</strong> ${fa.exercise_identified}</span>
        <span><strong>Input Duration:</strong> ${fa.evidence_basis.duration_sec}s (${fa.evidence_basis.total_frames} frames @ 100 Hz)</span>
        <span><strong>Ground Truth:</strong> ${fa.evidence_basis.has_ground_truth ? '✓ Synchronized Optical Reference Available' : '⚠ Sensor Only (No Reference)'}</span>
      </div>
      <p style="margin:0.25rem 0 0 0; font-style:italic;">
        <strong>Clinical Evidence Notice:</strong> ${fa.clinical_disclaimer}
      </p>
    </div>
  `;
}
