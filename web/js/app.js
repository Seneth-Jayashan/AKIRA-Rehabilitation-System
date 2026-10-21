/**
 * AKIRA Rehabilitation System - Frontend Controller
 * Connects to Python REST Backend, renders charts, handles live inference playground,
 * and controls batch verification.
 */

const API_BASE = window.location.origin;

// State
let appState = {
  currentTab: 'overview',
  activeModel: 'xgboost_opt',
  modelsData: null,
  overviewData: null,
  presetSamples: {},
  currentFeatures: {},
  currentTrueLabel: null,
  activeConfusionMatrix: 'xgboost',
  chartInstances: {}
};

// Activity metadata
const ACTIVITY_META = {
  'Walking': { icon: '🚶', color: '#38bdf8' },
  'Heel-Toe Gait': { icon: '👟', color: '#10b981' },
  'Hip Abduction': { icon: '🦵', color: '#818cf8' },
  'Knee Flexion': { icon: '🦿', color: '#f59e0b' },
  'Squat': { icon: '🏋️', color: '#a855f7' },
  'Abnormal Gait': { icon: '⚠️', color: '#f43f5e' }
};

// Confusion matrix descriptions
const CM_DESCRIPTIONS = {
  'xgboost': {
    title: 'XGBoost (Optimized & Balanced)',
    img: '/results/confusion_matrix_xgboost.png',
    desc: 'The optimized XGBoost model utilizes gradient-boosted decision trees with balanced class weights. It demonstrates superior recognition across minority classes (Squats, Knee Flexion) while maintaining a balanced Macro-F1 of 20.89%.'
  },
  'random_forest': {
    title: 'Random Forest (Baseline Champion)',
    img: '/results/confusion_matrix_random_forest_(baseline).png',
    desc: 'Achieved the highest overall raw accuracy (64.59%). Decision trees leverage orthogonal feature splits on frequency entropy and variance. Shows exceptional precision on the majority Walking and Heel-Toe classes.'
  },
  '1d_cnn': {
    title: '1D Convolutional Neural Network',
    img: '/results/confusion_matrix_1d_cnn.png',
    desc: 'Evaluated on raw temporal sequences (6 channels x 200 samples). Accuracy reached 42.02%. The 1D kernels capture local kinematic wave patterns but struggle with global frequency distribution compared to explicit Welch PSD features.'
  },
  'lstm': {
    title: 'Long Short-Term Memory (LSTM)',
    img: '/results/confusion_matrix_lstm.png',
    desc: 'Recurrent architecture designed to model temporal dependencies across time. Achieved 33.75% accuracy. Suffered from high computational latency (2.3s per batch) without surpassing tree-based accuracy.'
  },
  'cnn_lstm': {
    title: 'CNN + LSTM Hybrid Network',
    img: '/results/confusion_matrix_cnn_lstm.png',
    desc: 'Combines spatial convolution feature extraction with recurrent sequence aggregation. Accuracy: 35.82%. Prone to gradient dispersion on segmented 2-second windows.'
  },
  'transformer': {
    title: 'Time-Series Transformer (Attention)',
    img: '/results/confusion_matrix_transformer.png',
    desc: 'Employs multi-head self-attention mechanisms over raw sensor windows. Reached 47.82% accuracy, but required over 18 minutes of training time and 3.87 seconds inference latency—making it unsuitable for edge mobile devices.'
  }
};

// Initialize Application
document.addEventListener('DOMContentLoaded', async () => {
  setupNavigation();
  setupEventListeners();
  await loadInitialData();
  renderTimelineStages();
  renderDatasetTable();
  initCharts();
  loadDefaultPreset('Walking');
});

// Setup Tab Navigation
function setupNavigation() {
  const tabs = document.querySelectorAll('.tab-btn');
  tabs.forEach(tab => {
    tab.addEventListener('click', (e) => {
      const target = tab.dataset.tab;
      switchTab(target);
    });
  });

  // Handle URL Hash if present
  const hash = window.location.hash.replace('#', '');
  if (['overview', 'benchmarks', 'ble-stream', 'testing', 'batch', 'science'].includes(hash)) {
    switchTab(hash);
  }
}

function switchTab(tabId) {
  appState.currentTab = tabId;
  window.location.hash = tabId;

  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.tab === tabId);
  });

  document.querySelectorAll('.tab-pane').forEach(pane => {
    pane.classList.toggle('active', pane.id === `tab-${tabId}`);
  });

  // Trigger chart resize if entering benchmarks
  if (tabId === 'benchmarks') {
    setTimeout(() => {
      Object.values(appState.chartInstances).forEach(chart => chart && chart.resize());
    }, 100);
  }

  // Trigger oscilloscope resize if entering ble-stream
  if (tabId === 'ble-stream') {
    setTimeout(() => {
      const canvas = document.getElementById('canvas-oscilloscope');
      if (canvas && canvas.parentElement) {
        canvas.width = canvas.parentElement.clientWidth;
        canvas.height = canvas.parentElement.clientHeight || 220;
      }
    }, 100);
  }
}

// Event Listeners for UI Controls
function setupEventListeners() {
  // Preset buttons
  document.querySelectorAll('.preset-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const activity = btn.dataset.activity;
      loadDefaultPreset(activity);
    });
  });

  // Random sample button
  const randomBtn = document.getElementById('btn-random-sample');
  if (randomBtn) {
    randomBtn.addEventListener('click', fetchRandomSample);
  }

  // Run Inference button
  const runBtn = document.getElementById('btn-run-inference');
  if (runBtn) {
    runBtn.addEventListener('click', runLiveInference);
  }

  // Model Selector in Testing
  const modelSelect = document.getElementById('playground-model-select');
  if (modelSelect) {
    modelSelect.addEventListener('change', (e) => {
      appState.activeModel = e.target.value;
      runLiveInference();
    });
  }

  // Sliders fine-tuning
  const sliders = document.querySelectorAll('.range-slider');
  sliders.forEach(slider => {
    slider.addEventListener('input', (e) => {
      const feature = slider.dataset.feature;
      const val = parseFloat(slider.value);
      const valDisplay = document.getElementById(`val-${feature}`);
      if (valDisplay) valDisplay.textContent = val.toFixed(2);
      appState.currentFeatures[feature] = val;
    });
  });

  // Confusion Matrix Selector Tabs
  document.querySelectorAll('.cm-select-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.cm-select-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      renderConfusionMatrix(btn.dataset.model);
    });
  });

  // Batch Test button
  const batchRunBtn = document.getElementById('btn-run-batch');
  if (batchRunBtn) {
    batchRunBtn.addEventListener('click', runBatchTest);
  }
}

// Fetch Initial Data from Backend
async function loadInitialData() {
  try {
    const [statusRes, overviewRes, modelsRes, samplesRes] = await Promise.all([
      fetch(`${API_BASE}/api/status`).then(r => r.json()),
      fetch(`${API_BASE}/api/overview`).then(r => r.json()),
      fetch(`${API_BASE}/api/models`).then(r => r.json()),
      fetch(`${API_BASE}/api/samples`).then(r => r.json())
    ]);

    appState.overviewData = overviewRes;
    appState.modelsData = modelsRes;
    appState.presetSamples = samplesRes.samples || {};

    // Update Status Pill
    const statusPill = document.getElementById('server-status-pill');
    if (statusPill) {
      statusPill.innerHTML = `<span class="status-indicator"></span> AI Online (${statusRes.active_models.length} Models Ready)`;
    }

    // Populate Benchmark Table
    renderBenchmarkTable(modelsRes.summary_table);
    renderStabilityRegressors(modelsRes.stability_regressors);

  } catch (err) {
    console.error('Error loading initial data:', err);
  }
}

// Render Timeline Stages (Tab 1: What We Did)
function renderTimelineStages() {
  const container = document.getElementById('timeline-container');
  if (!container || !appState.overviewData) return;

  const stages = appState.overviewData.stages || [];
  container.innerHTML = stages.map((st, idx) => `
    <div class="stage-card">
      <div class="stage-header">
        <div class="stage-title-wrap">
          <div class="stage-number">${idx + 1}</div>
          <div>
            <h4>${st.title}</h4>
            <span class="stage-meta">${st.stage} • ${st.phase}</span>
          </div>
        </div>
        <span class="stage-badge">${st.badge}</span>
      </div>
      <p class="stage-desc">${st.description}</p>
      <ul class="stage-bullets">
        ${st.highlights.map(h => `<li>${h}</li>`).join('')}
      </ul>
    </div>
  `).join('');
}

// Render Dataset Table (Tab 1)
function renderDatasetTable() {
  const tbody = document.getElementById('dataset-table-body');
  if (!tbody || !appState.overviewData) return;

  const datasets = appState.overviewData.dataset_roles || [];
  tbody.innerHTML = datasets.map(d => `
    <tr>
      <td><strong>${d.dataset}</strong></td>
      <td><span class="table-tag">${d.type}</span></td>
      <td><strong>${d.role}</strong></td>
      <td>${d.use_case}</td>
    </tr>
  `).join('');
}

// Render Benchmark Table (Tab 2)
function renderBenchmarkTable(models) {
  const tbody = document.getElementById('benchmark-table-body');
  if (!tbody || !models) return;

  tbody.innerHTML = models.map(m => `
    <tr>
      <td>
        <strong>${m.name}</strong><br>
        <span style="font-size:0.75rem; color:var(--text-dim);">${m.type}</span>
      </td>
      <td><span class="table-tag">${m.track}</span></td>
      <td><strong style="color:var(--emerald-primary);">${m.accuracy.toFixed(2)}%</strong></td>
      <td><strong style="color:var(--cyan-primary);">${m.macro_f1.toFixed(2)}%</strong></td>
      <td>${m.train_time_sec}s</td>
      <td><span style="font-family:var(--font-mono);">${m.inference_latency_ms} ms</span></td>
      <td><span class="status-badge" style="font-size:0.72rem;">${m.badge}</span></td>
    </tr>
  `).join('');
}

// Render Stability Regressors
function renderStabilityRegressors(regressors) {
  const container = document.getElementById('stability-regressors-list');
  if (!container || !regressors) return;

  container.innerHTML = regressors.map(r => `
    <div style="display:flex; justify-content:space-between; align-items:center; padding:12px; background:rgba(255,255,255,0.02); border-radius:var(--radius-sm); margin-bottom:8px; border:1px solid rgba(255,255,255,0.05);">
      <div>
        <strong style="font-size:0.9rem;">${r.model}</strong><br>
        <span style="font-size:0.75rem; color:var(--text-dim);">${r.status}</span>
      </div>
      <div style="text-align:right;">
        <span style="font-family:var(--font-mono); font-weight:700; color:var(--emerald-primary);">R² = ${r.r2.toFixed(4)}</span><br>
        <span style="font-size:0.72rem; color:var(--text-muted);">RMSE = ${r.rmse}</span>
      </div>
    </div>
  `).join('');
}

// Confusion Matrix View Switcher
function renderConfusionMatrix(modelKey) {
  appState.activeConfusionMatrix = modelKey;
  const data = CM_DESCRIPTIONS[modelKey];
  if (!data) return;

  const imgEl = document.getElementById('cm-image');
  const titleEl = document.getElementById('cm-title');
  const descEl = document.getElementById('cm-description');

  if (imgEl) imgEl.src = data.img;
  if (titleEl) titleEl.textContent = data.title;
  if (descEl) descEl.textContent = data.desc;
}

// Initialize Charts using Chart.js
function initCharts() {
  if (typeof Chart === 'undefined') {
    console.warn('Chart.js not yet loaded.');
    return;
  }

  // 1. Accuracy vs Macro-F1 Chart
  const accCtx = document.getElementById('chart-accuracy-comparison');
  if (accCtx && appState.modelsData) {
    const models = appState.modelsData.summary_table || [];
    appState.chartInstances.accuracy = new Chart(accCtx, {
      type: 'bar',
      data: {
        labels: models.map(m => m.name.split(' (')[0]),
        datasets: [
          {
            label: 'Accuracy (%)',
            data: models.map(m => m.accuracy),
            backgroundColor: 'rgba(56, 189, 248, 0.8)',
            borderRadius: 6
          },
          {
            label: 'Macro F1 (%)',
            data: models.map(m => m.macro_f1),
            backgroundColor: 'rgba(99, 102, 241, 0.8)',
            borderRadius: 6
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: '#94a3b8', font: { family: 'Inter' } } }
        },
        scales: {
          x: { ticks: { color: '#94a3b8', font: { size: 10 } }, grid: { display: false } },
          y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255, 255, 255, 0.05)' }, max: 100 }
        }
      }
    });
  }

  // 2. Latency vs Train Time Chart
  const latCtx = document.getElementById('chart-latency-comparison');
  if (latCtx && appState.modelsData) {
    const models = appState.modelsData.summary_table || [];
    appState.chartInstances.latency = new Chart(latCtx, {
      type: 'bar',
      data: {
        labels: models.map(m => m.name.split(' (')[0]),
        datasets: [
          {
            label: 'Inference Latency per sample (ms)',
            data: models.map(m => m.inference_latency_ms),
            backgroundColor: 'rgba(16, 185, 129, 0.8)',
            borderRadius: 6
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: '#94a3b8' } }
        },
        scales: {
          x: { ticks: { color: '#94a3b8', font: { size: 10 } }, grid: { display: false } },
          y: { 
            type: 'logarithmic',
            ticks: { color: '#94a3b8' },
            grid: { color: 'rgba(255, 255, 255, 0.05)' }
          }
        }
      }
    });
  }

  // 3. Feature Importance Top 10 Chart
  fetch(`${API_BASE}/api/feature-importance`)
    .then(r => r.json())
    .then(data => {
      const topFeatCtx = document.getElementById('chart-feature-importance');
      if (topFeatCtx && data.xgboost) {
        const feats = data.xgboost.slice(0, 8);
        appState.chartInstances.features = new Chart(topFeatCtx, {
          type: 'bar',
          data: {
            labels: feats.map(f => f.feature),
            datasets: [{
              label: 'XGBoost Gain',
              data: feats.map(f => f.importance),
              backgroundColor: 'rgba(56, 189, 248, 0.7)',
              borderRadius: 6
            }]
          },
          options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
              legend: { display: false }
            },
            scales: {
              x: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255, 255, 255, 0.05)' } },
              y: { ticks: { color: '#94a3b8', font: { family: 'JetBrains Mono', size: 11 } }, grid: { display: false } }
            }
          }
        });
      }
    });
}

// Preset Loader for Live Playground
function loadDefaultPreset(activityName) {
  // Update button active state
  document.querySelectorAll('.preset-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.activity === activityName);
  });

  const samplesList = appState.presetSamples[activityName] || [];
  if (samplesList.length > 0) {
    const sample = samplesList[0];
    appState.currentFeatures = { ...sample.features };
    appState.currentTrueLabel = sample.label;
    updateSlidersFromFeatures(sample.key_features);
    runLiveInference();
  }
}

// Update Sliders based on current feature values
function updateSlidersFromFeatures(keyFeatures) {
  if (!keyFeatures) return;
  for (const [key, val] of Object.entries(keyFeatures)) {
    const slider = document.querySelector(`.range-slider[data-feature="${key}"]`);
    const valDisplay = document.getElementById(`val-${key}`);
    if (slider) {
      slider.value = val;
    }
    if (valDisplay) {
      valDisplay.textContent = parseFloat(val).toFixed(2);
    }
    appState.currentFeatures[key] = val;
  }
}

// Fetch Random Sample from Test Set
async function fetchRandomSample() {
  try {
    const res = await fetch(`${API_BASE}/api/random-sample`).then(r => r.json());
    if (res && res.features) {
      appState.currentFeatures = { ...res.features };
      appState.currentTrueLabel = res.true_label;

      // Update Preset button highlighting
      document.querySelectorAll('.preset-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.activity === res.true_label);
      });

      updateSlidersFromFeatures(res.key_features);
      runLiveInference();
    }
  } catch (err) {
    console.error('Error fetching random sample:', err);
  }
}

// Run Live Inference on Backend
async function runLiveInference() {
  const model = appState.activeModel || 'xgboost_opt';
  const features = appState.currentFeatures;

  try {
    const res = await fetch(`${API_BASE}/api/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model, features })
    }).then(r => r.json());

    if (res.error) {
      console.error(res.error);
      return;
    }

    renderPredictionOutput(res);
  } catch (err) {
    console.error('Prediction failed:', err);
  }
}

// Render Prediction Output to UI
function renderPredictionOutput(data) {
  const meta = ACTIVITY_META[data.predicted_class] || { icon: '🏃', color: '#38bdf8' };

  // 1. Icon & Names
  const iconEl = document.getElementById('pred-icon');
  const nameEl = document.getElementById('pred-name');
  const confEl = document.getElementById('pred-conf');
  const badgeEl = document.getElementById('pred-ground-truth-badge');
  const latencyEl = document.getElementById('pred-latency');

  if (iconEl) iconEl.textContent = meta.icon;
  if (nameEl) nameEl.textContent = data.predicted_class;
  if (confEl) confEl.innerHTML = `Confidence: <b>${data.confidence}%</b>`;
  if (latencyEl) latencyEl.textContent = `${data.latency_ms} ms`;

  // Ground Truth badge
  if (badgeEl) {
    if (appState.currentTrueLabel) {
      const isMatch = appState.currentTrueLabel === data.predicted_class;
      badgeEl.className = `ground-truth-badge ${isMatch ? 'match' : 'mismatch'}`;
      badgeEl.textContent = isMatch 
        ? `✓ Matches Ground Truth (${appState.currentTrueLabel})` 
        : `✗ Ground Truth: ${appState.currentTrueLabel}`;
      badgeEl.style.display = 'inline-block';
    } else {
      badgeEl.style.display = 'none';
    }
  }

  // 2. Probabilities Bars
  const probasList = document.getElementById('pred-probas-list');
  if (probasList && data.probabilities) {
    probasList.innerHTML = Object.entries(data.probabilities).map(([cName, pVal]) => `
      <div class="proba-item">
        <div class="proba-labels">
          <span class="name">${cName}</span>
          <span class="percent">${pVal.toFixed(1)}%</span>
        </div>
        <div class="proba-bar-bg">
          <div class="proba-bar-fill" style="width: ${pVal}%;"></div>
        </div>
      </div>
    `).join('');
  }

  // 3. Stability Index Meter
  const stabScoreEl = document.getElementById('pred-stability-score');
  const stabCatEl = document.getElementById('pred-stability-category');
  const stabDescEl = document.getElementById('pred-stability-desc');

  if (stabScoreEl) stabScoreEl.textContent = `${data.stability_score}%`;
  if (stabCatEl) {
    stabCatEl.textContent = data.stability_category;
    let colClass = 'status-green';
    if (data.stability_score < 70) colClass = 'status-amber';
    if (data.stability_score < 50) colClass = 'status-red';
    stabCatEl.className = `metric-sm-status ${colClass}`;
  }
  if (stabDescEl) stabDescEl.textContent = data.stability_description;

  // 4. Smoothness & Jerk
  const jerkEl = document.getElementById('pred-jerk-val');
  const smoothStatusEl = document.getElementById('pred-smoothness-status');

  if (jerkEl) jerkEl.textContent = `${data.smoothness.rms_jerk} m/s³`;
  if (smoothStatusEl) {
    smoothStatusEl.textContent = data.smoothness.status;
    let colClass = 'status-green';
    if (data.smoothness.rms_jerk > 6.5) colClass = 'status-amber';
    if (data.smoothness.rms_jerk > 12.0) colClass = 'status-red';
    smoothStatusEl.className = `metric-sm-status ${colClass}`;
  }

  // 5. Clinical Recommendations
  const clinicalNotesEl = document.getElementById('pred-clinical-notes');
  const rehabRecEl = document.getElementById('pred-rehab-rec');

  if (clinicalNotesEl) clinicalNotesEl.textContent = data.clinical_notes;
  if (rehabRecEl) rehabRecEl.textContent = data.recommendation;
}

// Run Batch Test & Evaluate Model
async function runBatchTest() {
  const model = document.getElementById('batch-model-select').value;
  const sampleCount = parseInt(document.getElementById('batch-sample-count').value, 10);
  const btn = document.getElementById('btn-run-batch');

  if (btn) {
    btn.disabled = true;
    btn.textContent = '⏳ Evaluating...';
  }

  try {
    const res = await fetch(`${API_BASE}/api/batch-test`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model, sample_count: sampleCount })
    }).then(r => r.json());

    if (res.error) {
      alert(res.error);
      return;
    }

    renderBatchResults(res);
  } catch (err) {
    console.error('Batch test error:', err);
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.textContent = 'Run Live Evaluation';
    }
  }
}

// Render Batch Evaluation Results
function renderBatchResults(data) {
  const container = document.getElementById('batch-results-container');
  if (!container) return;
  container.style.display = 'grid';

  // Stats Highlights
  document.getElementById('batch-acc-val').textContent = `${data.accuracy}%`;
  document.getElementById('batch-correct-val').textContent = `${data.correct_predictions} / ${data.sample_count}`;
  document.getElementById('batch-latency-val').textContent = `${data.avg_latency_per_window_ms} ms / window`;

  // Render Confusion Matrix Grid Table
  const table = document.getElementById('batch-matrix-table');
  if (table && data.confusion_matrix) {
    let html = '<thead><tr><th>True \\ Pred</th>';
    data.classes.forEach(c => {
      html += `<th>${c.split(' ')[0]}</th>`;
    });
    html += '</tr></thead><tbody>';

    data.confusion_matrix.forEach((row, rIdx) => {
      const trueClass = data.classes[rIdx];
      html += `<tr><th>${trueClass}</th>`;
      row.forEach((cellVal, cIdx) => {
        const isDiag = rIdx === cIdx;
        const cellClass = isDiag && cellVal > 0 ? 'correct' : (cellVal > 0 ? 'mismatch' : '');
        html += `<td class="${cellClass}">${cellVal}</td>`;
      });
      html += '</tr>';
    });
    html += '</tbody>';
    table.innerHTML = html;
  }

  // Render Class Breakdown Table
  const classTbody = document.getElementById('batch-class-stats-body');
  if (classTbody && data.class_stats) {
    classTbody.innerHTML = Object.entries(data.class_stats).map(([cName, stats]) => `
      <tr>
        <td><strong>${cName}</strong></td>
        <td>${stats.support}</td>
        <td>${stats.correct}</td>
        <td style="color:var(--cyan-primary); font-weight:600;">${stats.precision}%</td>
        <td style="color:var(--indigo-primary); font-weight:600;">${stats.recall}%</td>
        <td style="color:var(--emerald-primary); font-weight:700;">${stats.f1}%</td>
      </tr>
    `).join('');
  }
}
