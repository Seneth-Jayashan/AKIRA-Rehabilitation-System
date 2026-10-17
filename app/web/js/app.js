let currentModelKey = 'hybrid_1_cnn';
let currentSampleIndex = 0;
let testSamplesList = [];
let modelsList = [];
let kneeChart = null;
let imuChart = null;
let longitudinalChart = null;
let lastSessionOutputs = null;
let webSerialPort = null;
let showConfusionMatrix = true;

async function initDashboard() {
  await loadModels();
  await loadTestSamples();
  await loadModelMetrics(currentModelKey);
  await loadBenchmarkTable();
  await selectSample(0);
  await loadESP32Status();
  await loadLongitudinalData();
}

async function loadModels() {
  try {
    const res = await fetch('/api/models');
    modelsList = await res.json();
    const container = document.getElementById('model-cards-container');
    container.innerHTML = '';

    modelsList.forEach(m => {
      const card = document.createElement('div');
      card.id = `card-${m.key}`;
      card.className = `model-card p-4 rounded-xl border border-slate-200 cursor-pointer transition-all hover:border-blue-400 bg-white ${m.key === currentModelKey ? 'active' : ''}`;
      card.onclick = () => selectModel(m.key);

      card.innerHTML = `
        <div class="flex items-start justify-between gap-2">
          <h3 class="text-sm font-bold text-slate-900">${m.title}</h3>
          ${m.edge_ready ? '<span class="text-[10px] px-1.5 py-0.5 rounded bg-blue-100 text-blue-800 font-semibold shrink-0">Edge AI</span>' : ''}
        </div>
        <p class="text-[11px] font-medium text-slate-500 mt-0.5">${m.subtitle}</p>
        <p class="text-xs text-slate-600 mt-2 line-clamp-2">${m.description}</p>
        <div class="flex items-center justify-between text-[11px] text-slate-500 mt-3 pt-2.5 border-t border-slate-100">
          <span class="font-mono">${m.params.toLocaleString()} params</span>
          <span class="text-blue-600 font-semibold">${m.latency_ms}</span>
        </div>
      `;
      container.appendChild(card);
    });
  } catch (err) {
    console.error('Error loading models:', err);
  }
}

async function selectModel(key) {
  currentModelKey = key;
  document.querySelectorAll('.model-card').forEach(c => c.classList.remove('active'));
  const activeCard = document.getElementById(`card-${key}`);
  if (activeCard) activeCard.classList.add('active');

  const modelObj = modelsList.find(m => m.key === key);
  if (modelObj) {
    document.getElementById('active-model-badge').innerText = `Selected: ${modelObj.title}`;
  }

  await loadModelMetrics(key);
  await runInference();
  highlightTableRow(key);
}

async function loadModelMetrics(key) {
  try {
    const res = await fetch(`/api/metrics/${key}`);
    const data = await res.json();

    document.getElementById('kpi-exercise-acc').innerText = `${data.exercise_acc}%`;
    document.getElementById('kpi-exercise-f1').innerText = `Macro F1: ${data.exercise_f1}%`;
    document.getElementById('kpi-quality-acc').innerText = `${data.quality_acc}%`;
    document.getElementById('kpi-quality-f1').innerText = `Macro F1: ${data.quality_f1}%`;
    document.getElementById('kpi-combined-score').innerText = `${data.combined_score}%`;
    document.getElementById('kpi-params').innerText = `${data.params.toLocaleString()} params`;

    renderConfusionMatrix(data.cm_exercise, data.class_names, 'cm-exercise-grid', 'blue');
    renderConfusionMatrix(data.cm_quality, data.quality_names, 'cm-quality-grid', 'emerald');
  } catch (err) {
    console.error('Error loading metrics:', err);
  }
}

function renderConfusionMatrix(matrix, labels, containerId, color) {
  const container = document.getElementById(containerId);
  if (!container) return;

  let html = '<table class="w-full text-center text-xs border border-slate-200 bg-white rounded-lg overflow-hidden">';
  html += '<thead><tr class="bg-slate-100 text-slate-600 text-[10px] font-bold"><th class="p-1.5 text-left">True \ Pred</th>';
  labels.forEach(l => {
    html += `<th class="p-1.5">${l.split(' ')[0]}</th>`;
  });
  html += '</tr></thead><tbody>';

  const maxVal = Math.max(...matrix.flat());

  matrix.forEach((row, rIdx) => {
    html += `<tr class="border-t border-slate-100"><td class="p-1.5 text-left font-semibold text-slate-700 bg-slate-50 text-[11px]">${labels[rIdx].split(' ')[0]}</td>`;
    row.forEach((val, cIdx) => {
      const isDiag = (rIdx === cIdx);
      const intensity = maxVal > 0 ? (val / maxVal) : 0;
      const bg = isDiag
        ? (color === 'blue' ? `rgba(37, 99, 235, ${0.15 + intensity * 0.7})` : `rgba(16, 185, 129, ${0.15 + intensity * 0.7})`)
        : (val > 0 ? `rgba(239, 68, 68, ${0.1 + intensity * 0.4})` : '#f8fafc');
      const textColor = (intensity > 0.5) ? '#ffffff' : '#0f172a';

      html += `<td class="p-2 font-mono font-bold cm-cell" style="background-color: ${bg}; color: ${textColor}">${val}</td>`;
    });
    html += '</tr>';
  });
  html += '</tbody></table>';
  container.innerHTML = html;
}

function toggleConfusionMatrix() {
  showConfusionMatrix = !showConfusionMatrix;
  const c = document.getElementById('cm-container');
  c.style.display = showConfusionMatrix ? 'block' : 'none';
}

async function loadTestSamples() {
  try {
    const res = await fetch('/api/test_samples');
    testSamplesList = await res.json();
    populateSampleDropdown(testSamplesList);
  } catch (err) {
    console.error('Error loading test samples:', err);
  }
}

function populateSampleDropdown(samples) {
  const select = document.getElementById('sample-select');
  select.innerHTML = '';
  if (samples.length === 0) {
    const opt = document.createElement('option');
    opt.value = -1;
    opt.innerText = '⚠️ No trials match this filter combination. Try selecting "All Exercises".';
    select.appendChild(opt);
    return;
  }
  samples.forEach(s => {
    const opt = document.createElement('option');
    opt.value = s.index;
    opt.innerText = `[#${s.index + 1}] ${s.file_id} | ${s.exercise_name} | GT: ${s.quality_name.split(' ')[0]} | ROM: ${s.rom_deg}° (${s.source_dataset})`;
    select.appendChild(opt);
  });
  select.value = samples[0].index;
}

function applyFilters() {
  let ex = document.getElementById('filter-exercise').value;
  const q = parseInt(document.getElementById('filter-quality').value);
  const notice = document.getElementById('filter-notice');

  let filtered = testSamplesList.filter(s => {
    const matchEx = (ex === 'all' || s.exercise_name.toLowerCase() === ex.toLowerCase());
    const matchQ = (q === -1 || s.quality_grade === q);
    return matchEx && matchQ;
  });

  // Smart auto-coordination if 0 trials match:
  if (filtered.length === 0) {
    if (q === 1) {
      // Quality 1 only exists in Seated Knee Extension (14 trials)
      document.getElementById('filter-exercise').value = 'Seated Knee Extension';
      ex = 'Seated Knee Extension';
      filtered = testSamplesList.filter(s => s.quality_grade === 1);
      if (notice) {
        notice.innerHTML = `ℹ️ <b>Clinical Note:</b> "Needs Improvement" (Quadriceps Extension Lag) is specifically measured during <b>Seated Knee Extension</b>. Filter automatically adjusted to show all 14 trials.`;
        notice.classList.remove('hidden');
      }
    } else if (q === 2 && (ex === 'Step-Up' || ex === 'Walking')) {
      // Quality 2 exists in Mini Squat and Seated Extension
      document.getElementById('filter-exercise').value = 'all';
      ex = 'all';
      filtered = testSamplesList.filter(s => s.quality_grade === 2);
      if (notice) {
        notice.innerHTML = `ℹ️ <b>Clinical Note:</b> "Bad / Compensatory Fault" is available for <b>Mini Squat</b> and <b>Seated Knee Extension</b> (52 trials total). Filter adjusted to "All Exercises".`;
        notice.classList.remove('hidden');
      }
    } else if (notice) {
      notice.innerHTML = `⚠️ No trials found for this exact combination. Showing all matching trials for this quality tier.`;
      notice.classList.remove('hidden');
    }
  } else {
    if (notice) notice.classList.add('hidden');
  }

  populateSampleDropdown(filtered);
  if (filtered.length > 0) {
    selectSample(filtered[0].index);
  }
}

function onSampleChanged() {
  const val = parseInt(document.getElementById('sample-select').value);
  selectSample(val);
}

function pickRandomSample() {
  const select = document.getElementById('sample-select');
  const options = select.options;
  if (options.length === 0) return;
  const randIdx = Math.floor(Math.random() * options.length);
  select.selectedIndex = randIdx;
  selectSample(parseInt(select.value));
}

function prevSample() {
  const select = document.getElementById('sample-select');
  if (select.selectedIndex > 0) {
    select.selectedIndex--;
    selectSample(parseInt(select.value));
  }
}

function nextSample() {
  const select = document.getElementById('sample-select');
  if (select.selectedIndex < select.options.length - 1) {
    select.selectedIndex++;
    selectSample(parseInt(select.value));
  }
}

let currentSampleData = null;
let currentRawFilter = 'all';

async function selectSample(sampleIdx) {
  currentSampleIndex = sampleIdx;
  try {
    const res = await fetch(`/api/sample/${sampleIdx}`);
    const data = await res.json();
    currentSampleData = data;

    // Update Metadata Card
    document.getElementById('meta-file-id').innerText = data.file_id;
    document.getElementById('meta-subject').innerText = data.subject;
    document.getElementById('meta-dataset').innerText = data.source_dataset;
    document.getElementById('meta-gt-exercise').innerText = data.exercise_name;
    document.getElementById('meta-gt-quality').innerText = data.quality_name;

    // Render 12 Raw Sensor Channel Badges
    if (data.raw_channels_summary) renderRawChannelsGrid(data.raw_channels_summary);

    // Render Raw Numerical Telemetry Matrix Table
    if (data.raw_matrix_rows) renderRawMatrixTable(data.raw_matrix_rows);

    // Render 16 Features
    renderFeaturesGrid(data.features);

    // Render Waveform Charts
    renderKneeChart(data.time_axis, data.knee_angle);
    if (data.all_12_channels) {
      render12ChannelIMUChart(data.time_axis, data.all_12_channels, currentRawFilter);
    }

    // Run live inference on this sample
    await runInference();
  } catch (err) {
    console.error('Error selecting sample:', err);
  }
}

function renderFeaturesGrid(features) {
  const container = document.getElementById('features-grid');
  container.innerHTML = '';
  features.forEach(f => {
    const card = document.createElement('div');
    card.className = 'bg-slate-50 rounded-lg p-2 border border-slate-200';
    card.innerHTML = `
      <div class="text-[10px] text-slate-500 font-medium truncate" title="${f.name} - ${f.desc}">${f.name}</div>
      <div class="text-sm font-bold text-slate-800 font-mono mt-0.5">${f.value} <span class="text-[10px] text-slate-400 font-normal">${f.unit}</span></div>
    `;
    container.appendChild(card);
  });
}

async function runInference() {
  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model_key: currentModelKey,
        sample_index: currentSampleIndex
      })
    });
    const pred = await res.json();

    // Exercise Recognition Result
    const ex = pred.predicted_exercise;
    document.getElementById('res-exercise-name').innerText = ex.name;
    document.getElementById('res-exercise-conf').innerText = `${ex.confidence}%`;
    const badgeEx = document.getElementById('badge-class-match');
    if (ex.matches_ground_truth) {
      badgeEx.className = 'text-[11px] font-bold px-2 py-0.5 rounded-full bg-emerald-100 text-emerald-800 border border-emerald-200';
      badgeEx.innerText = '✅ Matches Ground Truth';
    } else {
      badgeEx.className = 'text-[11px] font-bold px-2 py-0.5 rounded-full bg-amber-100 text-amber-800 border border-amber-200';
      badgeEx.innerText = `⚠️ GT: ${ex.true_name}`;
    }
    renderProbBars('prob-bars-exercise', ex.class_names, ex.probabilities, ex.class_id, 'blue');

    // Quality Assessment Result
    const q = pred.predicted_quality;
    document.getElementById('res-quality-name').innerText = q.name;
    document.getElementById('res-quality-conf').innerText = `${q.confidence}%`;
    const badgeQ = document.getElementById('badge-qual-match');
    if (q.matches_ground_truth) {
      badgeQ.className = 'text-[11px] font-bold px-2 py-0.5 rounded-full bg-emerald-100 text-emerald-800 border border-emerald-200';
      badgeQ.innerText = '✅ Matches Ground Truth';
    } else {
      badgeQ.className = 'text-[11px] font-bold px-2 py-0.5 rounded-full bg-amber-100 text-amber-800 border border-amber-200';
      badgeQ.innerText = `⚠️ GT: ${q.true_name}`;
    }
    renderProbBars('prob-bars-quality', q.grade_names, q.probabilities, q.grade_id, 'emerald');

    // Update Live Pipeline Execution Latency Tracker
    if (pred.pipeline_latency) {
      const lat = pred.pipeline_latency;
      const elMath = document.getElementById('pipe-math-time');
      const elInfer = document.getElementById('pipe-infer-time');
      const elTotal = document.getElementById('pipe-total-time');
      if (elMath) elMath.innerText = `math_feature_extractor.py: ${lat.math_ms} ms`;
      if (elInfer) elInfer.innerText = `Active AI Inference: ${lat.ai_ms} ms`;
      if (elTotal) elTotal.innerText = `⚡ Live End-to-End Latency: ${lat.total_ms} ms`;
      const badgeMath = document.getElementById('features-badge-math');
      if (badgeMath) badgeMath.innerText = `⚡ Computed On-The-Fly in ${lat.math_ms} ms via math_feature_extractor.py`;
    }

    // Populate Box 07 (Official 7 Session-Level Outputs)
    if (pred.session_outputs) {
      updateBox07UI(pred.session_outputs);
    }

    // Clinical feedback list
    const feedbackList = document.getElementById('clinical-feedback-list');
    feedbackList.innerHTML = '';
    pred.clinical_feedback.forEach(fb => {
      const li = document.createElement('li');
      li.innerText = fb;
      feedbackList.appendChild(li);
    });

  } catch (err) {
    console.error('Error running inference:', err);
  }
}

function renderProbBars(containerId, names, probs, activeIdx, color) {
  const container = document.getElementById(containerId);
  container.innerHTML = '';
  names.forEach((name, idx) => {
    const prob = probs[idx];
    const isWinner = (idx === activeIdx);
    const row = document.createElement('div');
    row.className = 'text-xs';
    const barColor = isWinner
      ? (color === 'blue' ? 'bg-blue-600' : 'bg-emerald-600')
      : 'bg-slate-300';

    row.innerHTML = `
      <div class="flex justify-between text-[11px] mb-0.5 ${isWinner ? 'font-bold text-slate-900' : 'text-slate-500'}">
        <span>${name}</span>
        <span class="font-mono">${prob}%</span>
      </div>
      <div class="w-full bg-slate-200 rounded-full h-1.5 overflow-hidden">
        <div class="${barColor} h-1.5 rounded-full transition-all duration-300" style="width: ${prob}%"></div>
      </div>
    `;
    container.appendChild(row);
  });
}

function renderKneeChart(timeAxis, kneeAngle) {
  const ctx = document.getElementById('kneeAngleChart').getContext('2d');
  if (kneeChart) kneeChart.destroy();

  kneeChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: timeAxis,
      datasets: [{
        label: 'Knee Flexion Angle (°)',
        data: kneeAngle,
        borderColor: '#2563eb',
        backgroundColor: 'rgba(37, 99, 235, 0.08)',
        fill: true,
        tension: 0.3,
        borderWidth: 2.2,
        pointRadius: 0
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            title: (items) => `Time: ${items[0].label}s`,
            label: (item) => `Knee Angle: ${item.raw}°`
          }
        }
      },
      scales: {
        x: {
          title: { display: true, text: 'Time (seconds)', font: { size: 10 } },
          ticks: { maxTicksLimit: 8, font: { size: 10 } }
        },
        y: {
          title: { display: true, text: 'Flexion Angle (°)', font: { size: 10 } },
          ticks: { font: { size: 10 } }
        }
      }
    }
  });
}

function renderRawChannelsGrid(summary) {
  const container = document.getElementById('raw-channels-grid');
  if (!container || !summary) return;
  container.innerHTML = '';
  summary.forEach(ch => {
    const isThigh = ch.loc === 'Thigh';
    const card = document.createElement('div');
    card.className = `p-2 rounded-lg border text-xs ${isThigh ? 'bg-blue-50/70 border-blue-200' : 'bg-emerald-50/70 border-emerald-200'}`;
    card.innerHTML = `
      <div class="flex items-center justify-between text-[10px] font-bold ${isThigh ? 'text-blue-900' : 'text-emerald-900'}">
        <span>CH ${ch.id}: ${ch.type} ${ch.name.split(' ').slice(-2).join(' ')}</span>
        <span class="text-[9px] px-1 rounded ${isThigh ? 'bg-blue-200/80 text-blue-900' : 'bg-emerald-200/80 text-emerald-900'}">${ch.loc}</span>
      </div>
      <div class="font-mono font-bold text-slate-800 text-xs mt-1">
        ${ch.mean} <span class="text-[10px] font-normal text-slate-500">${ch.unit}</span>
      </div>
      <div class="flex justify-between text-[9px] text-slate-500 mt-0.5 font-mono">
        <span>Min: ${ch.min}</span>
        <span>Max: ${ch.max}</span>
      </div>
    `;
    container.appendChild(card);
  });
}

function renderRawMatrixTable(rows) {
  const tbody = document.getElementById('raw-data-tbody');
  if (!tbody || !rows) return;
  tbody.innerHTML = '';
  rows.forEach(r => {
    const tr = document.createElement('tr');
    tr.className = 'hover:bg-slate-50 transition-colors font-mono text-[10px]';
    let cols = `<td class="p-1 font-bold text-slate-700 bg-slate-100 border-r border-slate-200">${r.t.toFixed(2)}s</td>`;
    r.values.forEach((v, idx) => {
      const bgCol = idx < 3 ? 'bg-blue-50/20' : (idx < 6 ? 'bg-amber-50/20' : (idx < 9 ? 'bg-emerald-50/20' : 'bg-red-50/20'));
      cols += `<td class="p-1 border-r border-slate-100 ${bgCol}">${v}</td>`;
    });
    tr.innerHTML = cols;
    tbody.appendChild(tr);
  });
}

function switchRawView(mode) {
  const waveView = document.getElementById('raw-waveform-view');
  const matrixView = document.getElementById('raw-matrix-view');
  const btnWave = document.getElementById('btn-view-waveforms');
  const btnMatrix = document.getElementById('btn-view-matrix');
  const filterGroup = document.getElementById('waveform-filter-group');

  if (mode === 'waveforms') {
    waveView.classList.remove('hidden');
    matrixView.classList.add('hidden');
    filterGroup.classList.remove('hidden');
    btnWave.className = 'px-3 py-1 rounded-md text-xs font-bold bg-blue-600 text-white shadow-2xs transition-colors';
    btnMatrix.className = 'px-3 py-1 rounded-md text-xs font-bold bg-white text-slate-700 border border-slate-200 hover:bg-slate-50 transition-colors';
  } else {
    waveView.classList.add('hidden');
    matrixView.classList.remove('hidden');
    filterGroup.classList.add('hidden');
    btnMatrix.className = 'px-3 py-1 rounded-md text-xs font-bold bg-blue-600 text-white shadow-2xs transition-colors';
    btnWave.className = 'px-3 py-1 rounded-md text-xs font-bold bg-white text-slate-700 border border-slate-200 hover:bg-slate-50 transition-colors';
  }
}

function filterIMUPlot(filterType) {
  currentRawFilter = filterType;
  ['all', 'acc', 'gyro'].forEach(f => {
    const btn = document.getElementById(`wf-btn-${f}`);
    if (btn) {
      if (f === filterType) {
        btn.className = 'px-2 py-0.5 rounded bg-slate-800 text-white font-semibold';
      } else {
        btn.className = 'px-2 py-0.5 rounded bg-white border border-slate-200 text-slate-600 hover:bg-slate-50';
      }
    }
  });
  if (currentSampleData && currentSampleData.all_12_channels) {
    render12ChannelIMUChart(currentSampleData.time_axis, currentSampleData.all_12_channels, filterType);
  }
}

function render12ChannelIMUChart(timeAxis, allChannels, filterType = 'all') {
  const ctx = document.getElementById('imuSignalChart').getContext('2d');
  if (imuChart) imuChart.destroy();

  let selectedChannels = allChannels;
  if (filterType === 'acc') {
    selectedChannels = allChannels.filter(c => c.type === 'Acc');
  } else if (filterType === 'gyro') {
    selectedChannels = allChannels.filter(c => c.type === 'Gyro');
  }

  const datasets = selectedChannels.map(ch => ({
    label: `${ch.name} (${ch.unit})`,
    data: ch.data,
    borderColor: ch.color,
    borderWidth: 1.4,
    pointRadius: 0,
    yAxisID: ch.unit === 'g' ? 'y' : 'y1'
  }));

  imuChart = new Chart(ctx, {
    type: 'line',
    data: { labels: timeAxis, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'top',
          labels: { boxWidth: 10, font: { size: 9 }, padding: 6 }
        }
      },
      scales: {
        x: {
          title: { display: true, text: 'Time (seconds)', font: { size: 9 } },
          ticks: { maxTicksLimit: 8, font: { size: 9 } }
        },
        y: {
          type: 'linear',
          position: 'left',
          title: { display: true, text: 'Acceleration (g)', font: { size: 9 } },
          ticks: { font: { size: 9 } }
        },
        y1: {
          type: 'linear',
          position: 'right',
          grid: { drawOnChartArea: false },
          title: { display: true, text: 'Angular Velocity (deg/s)', font: { size: 9 } },
          ticks: { font: { size: 9 } }
        }
      }
    }
  });
}

async function loadBenchmarkTable() {
  try {
    const res = await fetch('/api/all_metrics');
    const rows = await res.json();
    const tbody = document.getElementById('benchmark-table-body');
    tbody.innerHTML = '';

    rows.forEach((r, idx) => {
      const tr = document.createElement('tr');
      tr.id = `row-${r.key}`;
      tr.className = `hover:bg-slate-50 transition-colors ${r.key === currentModelKey ? 'bg-blue-50/50 font-semibold' : ''}`;
      tr.innerHTML = `
        <td class="py-2.5 px-3 font-bold text-slate-700">#${idx + 1}</td>
        <td class="py-2.5 px-3 font-semibold text-slate-900">${r.title}</td>
        <td class="py-2.5 px-3 text-right font-mono text-slate-600">${r.params.toLocaleString()}</td>
        <td class="py-2.5 px-3 text-right font-mono font-bold text-blue-600">${r.exercise_acc}%</td>
        <td class="py-2.5 px-3 text-right font-mono text-slate-600">${r.exercise_f1}%</td>
        <td class="py-2.5 px-3 text-right font-mono font-bold text-emerald-600">${r.quality_acc}%</td>
        <td class="py-2.5 px-3 text-right font-mono text-slate-600">${r.quality_f1}%</td>
        <td class="py-2.5 px-3 text-right font-mono font-extrabold text-indigo-700">${r.combined_score}%</td>
        <td class="py-2.5 px-3 text-center">
          <button onclick="selectModel('${r.key}')" class="px-2.5 py-1 text-[11px] font-semibold bg-slate-100 hover:bg-blue-600 hover:text-white rounded text-slate-700 transition-colors">
            Select
          </button>
        </td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error('Error loading benchmark table:', err);
  }
}

function highlightTableRow(key) {
  document.querySelectorAll('#benchmark-table-body tr').forEach(tr => {
    tr.classList.remove('bg-blue-50/50', 'font-semibold');
  });
  const activeRow = document.getElementById(`row-${key}`);
  if (activeRow) {
    activeRow.classList.add('bg-blue-50/50', 'font-semibold');
  }
}

// ============================================================================
// BOX 07: POPULATE THE 7 OFFICIAL SESSION-LEVEL OUTPUTS
// ============================================================================
function updateBox07UI(so) {
  if (!so) return;
  lastSessionOutputs = so;

  const setEl = (id, txt) => {
    const el = document.getElementById(id);
    if (el) el.innerText = txt;
  };

  setEl('box07-ex', so.out_01_recognized_exercise);
  setEl('box07-ex-conf', `Conf: ${so.out_01_confidence_pct}%`);
  setEl('box07-rom', `${so.out_02_rom_deg}°`);
  setEl('box07-max-angle', `Max θ: ${so.out_02_max_knee_angle_deg}°`);
  setEl('box07-depth', `${so.out_03_movement_depth_deg}°`);
  setEl('box07-lag', `Quad Lag: ${so.out_03_extension_lag_deg}°`);
  setEl('box07-reps', `${so.out_04_repetition_count} rep(s)`);
  setEl('box07-smooth', `${so.out_05_smoothness_norm}`);
  setEl('box07-sparc', `SPARC: ${so.out_05_sparc_raw}`);
  setEl('box07-dur', `${so.out_06_duration_sec} s`);
  setEl('box07-vel', `Peak ω: ${so.out_06_peak_velocity_deg_s}°/s`);
  setEl('box07-qscore', `${so.out_07_quality_score_100} / 100`);
  setEl('box07-qgrade', so.out_07_quality_grade_name.split(' ')[0]);
}

// ============================================================================
// OPTION 1: LIVE ESP32 DUAL-IMU SENSOR RECORDER & DATASET BUILDER
// ============================================================================
function switchInputSourceMode(mode) {
  const espPanel = document.getElementById('esp32-recorder-panel');
  const testPanel = document.getElementById('testset-selector-panel');
  const btnTest = document.getElementById('tab-btn-testset');
  const btnEsp = document.getElementById('tab-btn-esp32');

  if (mode === 'esp32') {
    espPanel.classList.remove('hidden');
    testPanel.classList.add('hidden');
    btnEsp.className = 'px-3.5 py-1.5 text-xs font-bold bg-rose-600 text-white rounded-lg shadow-sm ring-2 ring-rose-300 transition-colors flex items-center gap-1.5';
    btnTest.className = 'px-3.5 py-1.5 text-xs font-bold bg-slate-100 text-slate-700 border border-slate-200 rounded-lg transition-colors';
    loadESP32Status();
  } else {
    espPanel.classList.add('hidden');
    testPanel.classList.remove('hidden');
    btnTest.className = 'px-3.5 py-1.5 text-xs font-bold bg-blue-600 text-white rounded-lg shadow-sm transition-colors';
    btnEsp.className = 'px-3.5 py-1.5 text-xs font-bold bg-rose-600 hover:bg-rose-700 text-white rounded-lg shadow-sm transition-colors flex items-center gap-1.5';
  }
}

async function loadESP32Status() {
  try {
    const res = await fetch('/api/esp32/status');
    const data = await res.json();
    const countEl = document.getElementById('esp32-csv-count');
    if (countEl) countEl.innerText = data.recorded_files_count;
  } catch (err) {
    console.error('Error loading ESP32 status:', err);
  }
}

async function connectWebSerialESP32() {
  const statusEl = document.getElementById('esp32-record-status');
  if (!('serial' in navigator)) {
    if (statusEl) statusEl.innerText = 'ℹ️ Using Backend Serial / Live Wi-Fi Stream Mode';
    return;
  }
  try {
    webSerialPort = await navigator.serial.requestPort();
    await webSerialPort.open({ baudRate: 115200 });
    if (statusEl) statusEl.innerText = '✅ Connected to ESP32 via Browser USB/BLE Serial (115200 baud)';
  } catch (e) {
    if (statusEl) statusEl.innerText = '📡 Ready in Backend Dual-IMU Stream Mode';
  }
}

async function startESP32Recording() {
  const btn = document.getElementById('btn-record-esp32');
  const statusEl = document.getElementById('esp32-record-status');
  const subjectId = document.getElementById('esp32-subject-id').value || 'P01_ORIF';
  const exerciseCode = document.getElementById('esp32-exercise').value;
  const qualityGrade = parseInt(document.getElementById('esp32-quality').value);
  const connMode = document.getElementById('esp32-conn-mode').value;

  if (btn) {
    btn.disabled = true;
    btn.innerHTML = '<span>⏳</span> Recording 200 Samples @ 100 Hz...';
  }
  if (statusEl) statusEl.innerText = '🔴 Sampling Thigh IMU (0x68) & Shin IMU (0x69) @ 100 Hz...';

  try {
    const res = await fetch('/api/esp32/record_and_analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        subject_id: subjectId,
        exercise_code: exerciseCode,
        quality_grade: qualityGrade,
        connection_mode: connMode,
        model_key: currentModelKey
      })
    });
    const out = await res.json();
    const data = out.sample_detail;
    const pred = out.prediction;
    currentSampleData = data;

    // Update Metadata Card
    document.getElementById('meta-file-id').innerText = data.file_id;
    document.getElementById('meta-subject').innerText = data.subject;
    document.getElementById('meta-dataset').innerText = data.source_dataset;
    document.getElementById('meta-gt-exercise').innerText = data.exercise_name;
    document.getElementById('meta-gt-quality').innerText = data.quality_name;

    // Stage 1: Raw 12-Channel Sensor Stream
    if (data.raw_channels_summary) renderRawChannelsGrid(data.raw_channels_summary);
    if (data.raw_matrix_rows) renderRawMatrixTable(data.raw_matrix_rows);
    if (data.all_12_channels) render12ChannelIMUChart(data.time_axis, data.all_12_channels, currentRawFilter);

    // Stage 2: 16 Clinical Features & Knee Angle Curve
    renderFeaturesGrid(data.features);
    renderKneeChart(data.time_axis, data.knee_angle);

    // Stage 3: AI Predictions & Box 07 (7 Session Outputs)
    const ex = pred.predicted_exercise;
    document.getElementById('res-exercise-name').innerText = ex.name;
    document.getElementById('res-exercise-conf').innerText = `${ex.confidence}%`;
    const badgeEx = document.getElementById('badge-class-match');
    badgeEx.className = 'text-[11px] font-bold px-2 py-0.5 rounded-full bg-emerald-100 text-emerald-800 border border-emerald-200';
    badgeEx.innerText = `✅ Recorded: ${ex.true_name}`;
    renderProbBars('prob-bars-exercise', ex.class_names, ex.probabilities, ex.class_id, 'blue');

    const q = pred.predicted_quality;
    document.getElementById('res-quality-name').innerText = q.name;
    document.getElementById('res-quality-conf').innerText = `${q.confidence}%`;
    const badgeQ = document.getElementById('badge-qual-match');
    badgeQ.className = 'text-[11px] font-bold px-2 py-0.5 rounded-full bg-emerald-100 text-emerald-800 border border-emerald-200';
    badgeQ.innerText = `✅ Label: ${q.true_name.split(' ')[0]}`;
    renderProbBars('prob-bars-quality', q.grade_names, q.probabilities, q.grade_id, 'emerald');

    if (pred.session_outputs) {
      updateBox07UI(pred.session_outputs);
    }

    const feedbackList = document.getElementById('clinical-feedback-list');
    feedbackList.innerHTML = '';
    pred.clinical_feedback.forEach(fb => {
      const li = document.createElement('li');
      li.innerText = fb;
      feedbackList.appendChild(li);
    });

    if (statusEl) statusEl.innerText = `✅ Saved ${out.saved_csv_filename}`;
    await loadESP32Status();
  } catch (err) {
    console.error('Error recording from ESP32:', err);
    if (statusEl) statusEl.innerText = '⚠️ Error during recording';
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = '<span>🔴</span> Start Recording (2.0s)';
    }
  }
}

async function openFirmwareModal() {
  const modal = document.getElementById('firmware-modal');
  const codeBox = document.getElementById('firmware-code-box');
  modal.classList.remove('hidden');
  modal.classList.add('flex');
  try {
    const res = await fetch('/api/esp32/firmware');
    const data = await res.json();
    codeBox.innerText = data.code;
  } catch (e) {
    codeBox.innerText = '// Error loading firmware file';
  }
}

function closeFirmwareModal() {
  const modal = document.getElementById('firmware-modal');
  modal.classList.add('hidden');
  modal.classList.remove('flex');
}

function copyFirmwareCode() {
  const code = document.getElementById('firmware-code-box').innerText;
  navigator.clipboard.writeText(code);
}

// ============================================================================
// BOX 08 & BOX 09: LONGITUDINAL TRACKING & RECOVERY PREDICTION ENGINE
// ============================================================================
async function loadLongitudinalData() {
  const targetRom = parseFloat(document.getElementById('clin-target-rom')?.value || 120);
  const safeLimit = parseFloat(document.getElementById('clin-safe-limit')?.value || 125);
  const freq = parseFloat(document.getElementById('clin-freq')?.value || 2);

  try {
    const res = await fetch(`/api/longitudinal?target_rom_deg=${targetRom}&max_safe_flexion_deg=${safeLimit}&sessions_per_week=${freq}`);
    const data = await res.json();
    renderLongitudinalUI(data);
  } catch (err) {
    console.error('Error loading longitudinal data:', err);
  }
}

async function saveCurrentSessionToLongitudinal() {
  if (!lastSessionOutputs) return;
  const targetRom = parseFloat(document.getElementById('clin-target-rom')?.value || 120);
  const safeLimit = parseFloat(document.getElementById('clin-safe-limit')?.value || 125);
  const freq = parseFloat(document.getElementById('clin-freq')?.value || 2);

  try {
    const res = await fetch('/api/longitudinal/add_session', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        exercise: lastSessionOutputs.out_01_recognized_exercise,
        rom_deg: lastSessionOutputs.out_02_rom_deg,
        depth_deg: lastSessionOutputs.out_03_movement_depth_deg,
        smoothness: lastSessionOutputs.out_05_smoothness_norm,
        quality_score: lastSessionOutputs.out_07_quality_score_100,
        rep_count: lastSessionOutputs.out_04_repetition_count,
        duration_s: lastSessionOutputs.out_06_duration_sec,
        target_rom_deg: targetRom,
        max_safe_flexion_deg: safeLimit,
        sessions_per_week: freq
      })
    });
    const data = await res.json();
    renderLongitudinalUI(data);
    document.getElementById('longitudinal-section')?.scrollIntoView({ behavior: 'smooth' });
  } catch (err) {
    console.error('Error saving session to longitudinal history:', err);
  }
}

async function resetLongitudinalHistory() {
  const targetRom = parseFloat(document.getElementById('clin-target-rom')?.value || 120);
  const safeLimit = parseFloat(document.getElementById('clin-safe-limit')?.value || 125);
  const freq = parseFloat(document.getElementById('clin-freq')?.value || 2);

  try {
    const res = await fetch(`/api/longitudinal/reset?target_rom_deg=${targetRom}&max_safe_flexion_deg=${safeLimit}&sessions_per_week=${freq}`, {
      method: 'POST'
    });
    const data = await res.json();
    renderLongitudinalUI(data);
  } catch (err) {
    console.error('Error resetting longitudinal history:', err);
  }
}

function renderLongitudinalUI(data) {
  const tr = data.trend_analysis;
  const mp = data.milestone_prediction;

  // Update 4 Trend KPIs
  document.getElementById('long-rom-rate').innerText = `${tr.rom_rate_per_session >= 0 ? '+' : ''}${tr.rom_rate_per_session}° / session`;
  document.getElementById('long-rom-total').innerText = `Total Gain: ${tr.total_rom_gain_deg >= 0 ? '+' : ''}${tr.total_rom_gain_deg}° (S1: ${tr.initial_rom_deg}° → Now: ${tr.current_rom_deg}°)`;
  document.getElementById('long-smooth-rate').innerText = `${tr.smoothness_rate_per_session >= 0 ? '+' : ''}${tr.smoothness_rate_per_session} / session`;
  document.getElementById('long-qual-rate').innerText = `${tr.quality_rate_per_session >= 0 ? '+' : ''}${tr.quality_rate_per_session} pts / session`;
  document.getElementById('long-r2').innerText = `R² = ${tr.r2_regression_fit}`;

  // Update Milestone Countdown Badge
  const estEl = document.getElementById('long-est-time');
  if (mp.sessions_to_milestone === 0) {
    estEl.innerText = `🎉 Target (${mp.target_rom_deg}°) Achieved!`;
  } else if (mp.sessions_to_milestone > 0) {
    estEl.innerText = `~${mp.sessions_to_milestone} Sessions (~${mp.days_to_milestone} Days)`;
  } else {
    estEl.innerText = `Plateau — Adjust Protocol`;
  }

  document.getElementById('long-status-banner').innerText = mp.milestone_status;

  // Render Box 08 Table (Actual Sessions + 3 Forecasted Sessions)
  const tbody = document.getElementById('longitudinal-table-body');
  tbody.innerHTML = '';

  data.sessions.forEach((s, idx) => {
    const row = document.createElement('tr');
    row.className = 'hover:bg-slate-50 transition-colors';
    const deltaStr = idx === 0 ? 'Baseline' : `${s.delta_rom >= 0 ? '+' : ''}${s.delta_rom}°`;
    const deltaCls = idx === 0 ? 'text-slate-400' : (s.delta_rom >= 0 ? 'text-emerald-600 font-bold' : 'text-rose-600 font-bold');

    row.innerHTML = `
      <td class="py-2 px-2.5 font-bold text-slate-900">
        ${s.session_id} <span class="text-[10px] font-normal text-slate-500 block">${s.exercise}</span>
      </td>
      <td class="py-2 px-2 text-right font-bold text-blue-700">${s.rom_deg}°</td>
      <td class="py-2 px-2 text-right text-[11px] ${deltaCls}">${deltaStr}</td>
      <td class="py-2 px-2 text-right text-slate-700">${s.depth_deg}°</td>
      <td class="py-2 px-2 text-right text-teal-700 font-semibold">${s.smoothness}</td>
      <td class="py-2 px-2.5 text-right font-extrabold text-emerald-700">${s.quality_score}</td>
    `;
    tbody.appendChild(row);
  });

  data.forecast_sessions.forEach(fs => {
    const row = document.createElement('tr');
    row.className = 'bg-indigo-50/40 text-indigo-900 italic';
    row.innerHTML = `
      <td class="py-1.5 px-2.5 font-bold text-indigo-700">🔮 ${fs.session_id}</td>
      <td class="py-1.5 px-2 text-right font-bold">${fs.rom_deg}°</td>
      <td class="py-1.5 px-2 text-right text-[10px] text-indigo-500">Predicted</td>
      <td class="py-1.5 px-2 text-right">${fs.depth_deg}°</td>
      <td class="py-1.5 px-2 text-right">${fs.smoothness}</td>
      <td class="py-1.5 px-2.5 text-right font-bold">${fs.quality_score}</td>
    `;
    tbody.appendChild(row);
  });

  // Render Box 09 Trajectory & Milestone Chart
  const ctx = document.getElementById('longitudinalChart').getContext('2d');
  if (longitudinalChart) longitudinalChart.destroy();

  const labels = [
    ...data.sessions.map(s => s.session_id),
    ...data.forecast_sessions.map(f => f.session_id.replace(' (Forecast)', '*'))
  ];
  const nActual = data.sessions.length;
  const totalLen = labels.length;

  const actualRomData = data.sessions.map(s => s.rom_deg).concat(Array(data.forecast_sessions.length).fill(null));
  const forecastRomData = Array(nActual - 1).fill(null)
    .concat([data.sessions[nActual - 1].rom_deg])
    .concat(data.forecast_sessions.map(f => f.rom_deg));

  const qualData = [
    ...data.sessions.map(s => s.quality_score),
    ...data.forecast_sessions.map(f => f.quality_score)
  ];
  const targetLine = Array(totalLen).fill(mp.target_rom_deg);

  longitudinalChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'Actual Recorded ROM (°)',
          data: actualRomData,
          borderColor: '#2563eb',
          backgroundColor: 'rgba(37, 99, 235, 0.12)',
          borderWidth: 2.8,
          pointRadius: 4,
          fill: true,
          tension: 0.25
        },
        {
          label: 'Predicted Trajectory ROM (°)',
          data: forecastRomData,
          borderColor: '#6366f1',
          borderDash: [6, 4],
          borderWidth: 2.5,
          pointRadius: 4,
          fill: false,
          tension: 0.25
        },
        {
          label: 'Session Quality Score (0–100)',
          data: qualData,
          borderColor: '#10b981',
          borderWidth: 2,
          pointRadius: 3,
          fill: false,
          tension: 0.25
        },
        {
          label: `Target Milestone (${mp.target_rom_deg}°)`,
          data: targetLine,
          borderColor: '#e11d48',
          borderDash: [4, 4],
          borderWidth: 1.8,
          pointRadius: 0,
          fill: false
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: 'top', labels: { boxWidth: 10, font: { size: 10 } } }
      },
      scales: {
        y: {
          min: 0,
          max: 140,
          title: { display: true, text: 'Degrees (°) / Quality Score (0–100)', font: { size: 10 } },
          ticks: { font: { size: 10 } }
        },
        x: {
          ticks: { font: { size: 10 } }
        }
      }
    }
  });
}

// Initialize on window load
window.addEventListener('DOMContentLoaded', initDashboard);
