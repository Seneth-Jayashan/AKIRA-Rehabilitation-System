/**
 * AKIRA Rehabilitation System - Live Web Bluetooth (BLE) IMU Controller & Real-Time Model Checker
 * 
 * Connects directly to ESP32 Dual-IMU Sensor Hub via Web Bluetooth API:
 * - Service UUID: 6E400001-B5A3-F393-E0A9-E50E24DCCA9E
 * - Data Char UUID: 6E400002-B5A3-F393-E0A9-E50E24DCCA9E (Notify, 52-byte DualImuPacket)
 * 
 * Features:
 * 1. Web Bluetooth BLE connection + packet stream decoding at 100Hz.
 * 2. High-fidelity hardware simulation engine for testing without physical board.
 * 3. 60 FPS Canvas Oscilloscope for 6-channel kinematic waveform monitoring.
 * 4. Dynamic Ankle Joint Kinematics (Range of Motion, Dorsiflexion/Plantarflexion, Rep Counter).
 * 5. Real-time 200-sample windowing & simultaneous multi-model evaluation (XGBoost, Random Forest, Transfer XGBoost, Stability RF).
 * 6. Live Accuracy Scoreboard & Ground Truth verification tracking.
 * 7. Trial session recording and standardized CSV export.
 */

// BLE Configuration matching ESP32 firmware (sketch_aug21a.ino)
const BLE_CONFIG = {
  SERVICE_UUID: '6e400001-b5a3-f393-e0a9-e50e24dcca9e',
  DATA_CHAR_UUID: '6e400002-b5a3-f393-e0a9-e50e24dcca9e',
  STATUS_CHAR_UUID: '6e400003-b5a3-f393-e0a9-e50e24dcca9e',
  DEVICE_NAME: 'ESP32-REHAB',
  PACKET_SIZE_BYTES: 52,
  SAMPLE_RATE_HZ: 100
};

// Activity Icons & Color Palette
const EXERCISE_ICONS = {
  'Walking': '🚶',
  'Heel-Toe Gait': '👟',
  'Squat': '🏋️',
  'Knee Flexion': '🦿',
  'Hip Abduction': '🦵',
  'Abnormal Gait': '⚠️'
};

// BLE State Controller
const bleState = {
  // Device & GATT
  device: null,
  server: null,
  dataChar: null,
  isConnected: false,

  // Simulation
  isSimulating: false,
  simTimer: null,
  simActivity: 'Walking',
  simPhase: 0,

  // Stream Buffers & Telemetry
  rollingBuffer: [],      // 200 samples (2.0s @ 100Hz)
  waveformBuffer: [],     // 300 samples for canvas oscilloscope
  packetCount: 0,
  rateCount: 0,
  rateHz: 0.0,
  lastRateCalcTime: performance.now(),
  lastInferenceTime: 0,
  inferenceIntervalMs: 800, // Trigger window inference every 800ms
  isInferencePending: false,

  // Kinematics & Ankle Range of Motion
  selectedChannel: 's1',  // 's1' (Tibia 0x68) or 's2' (Foot 0x69)
  currentAnkleAngle: 0.0,
  maxDorsiflexion: 0.0,
  maxPlantarflexion: 0.0,
  repCount: 0,
  lastZeroCrossTime: 0,
  repCadence: 0,
  gyHistory: [],

  // Live Ground Truth & Model Accuracy Verification
  activeGroundTruth: 'Walking',
  sessionStats: {
    totalWindows: 0,
    matchedWindows: 0,
    perModelHits: {
      'xgboost_opt': { total: 0, hits: 0 },
      'random_forest_opt': { total: 0, hits: 0 },
      'xgboost_transfer': { total: 0, hits: 0 }
    },
    confusionTally: {}
  },

  // Trial Session Recording
  recording: {
    isRecording: false,
    records: [],
    startTime: 0,
    participantId: 'P01'
  }
};

// Initialize Controller once DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  initBleControls();
  initOscilloscope();
});

/**
 * Setup Button Listeners and Interactive Controls
 */
function initBleControls() {
  const btnConnect = document.getElementById('btn-ble-connect');
  const btnDisconnect = document.getElementById('btn-ble-disconnect');
  const btnSimulate = document.getElementById('btn-ble-simulate');
  const channelSelect = document.getElementById('sensor-channel-select');
  const btnRecord = document.getElementById('btn-toggle-record');
  const btnExport = document.getElementById('btn-export-csv');

  if (btnConnect) btnConnect.addEventListener('click', connectWebBluetooth);
  if (btnDisconnect) btnDisconnect.addEventListener('click', disconnectWebBluetooth);
  if (btnSimulate) btnSimulate.addEventListener('click', toggleHardwareSimulation);

  if (channelSelect) {
    channelSelect.addEventListener('change', (e) => {
      bleState.selectedChannel = e.target.value;
      const lbl = document.getElementById('scope-channel-label');
      if (lbl) {
        lbl.textContent = e.target.value === 's1' ? 'Tibia Sensor (0x68 Proximal)' : 'Foot Sensor (0x69 Distal)';
      }
    });
  }

  // Simulation Activity Buttons
  document.querySelectorAll('.sim-act-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.sim-act-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      bleState.simActivity = btn.dataset.act;
      // Also sync active ground truth if user desires
      selectGroundTruth(btn.dataset.act);
    });
  });

  // Ground Truth Selector Buttons
  document.querySelectorAll('.gt-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      selectGroundTruth(btn.dataset.gt);
    });
  });

  // Recording Controls
  if (btnRecord) btnRecord.addEventListener('click', toggleTrialRecording);
  if (btnExport) btnExport.addEventListener('click', exportSessionCSV);
}

/**
 * Select Active Ground Truth Exercise for Live Accuracy Tracking
 */
function selectGroundTruth(activity) {
  bleState.activeGroundTruth = activity;
  document.querySelectorAll('.gt-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.gt === activity);
  });

  const bannerDesc = document.getElementById('live-match-reason');
  if (bannerDesc && !bleState.isConnected && !bleState.isSimulating) {
    bannerDesc.textContent = `Ground Truth target set to ${activity}. Connect sensor or start simulation to evaluate.`;
  }
}

/**
 * Web Bluetooth API Connection Handler
 */
async function connectWebBluetooth() {
  // Check browser support for Web Bluetooth API
  if (!navigator.bluetooth) {
    alert("Web Bluetooth API is not supported in this browser. Please use Google Chrome, Microsoft Edge, or Opera on Windows, macOS, Android, or Linux (with experimental Web Platform features enabled). You can also use 'Simulate Real BLE Stream' to test all models right now!");
    return;
  }

  try {
    updateBleStatusUI('connecting', 'Searching for ESP32-REHAB...');

    console.log('[BLE] Requesting Bluetooth Device...');
    let device;
    try {
      device = await navigator.bluetooth.requestDevice({
        filters: [
          { name: BLE_CONFIG.DEVICE_NAME },
          { services: [BLE_CONFIG.SERVICE_UUID] }
        ],
        optionalServices: [BLE_CONFIG.SERVICE_UUID]
      });
    } catch (filterErr) {
      console.warn('[BLE] Filtered search failed or cancelled, prompting with open device picker...', filterErr);
      device = await navigator.bluetooth.requestDevice({
        acceptAllDevices: true,
        optionalServices: [BLE_CONFIG.SERVICE_UUID]
      });
    }

    if (!device) {
      updateBleStatusUI('disconnected', 'Pairing cancelled.');
      return;
    }

    bleState.device = device;
    device.addEventListener('gattserverdisconnected', onBLEDisconnected);

    updateBleStatusUI('connecting', `Connecting to ${device.name || 'ESP32 IMU'} GATT Server...`);
    const server = await device.gatt.connect();
    bleState.server = server;

    console.log('[BLE] Getting Primary Service...');
    const service = await server.getPrimaryService(BLE_CONFIG.SERVICE_UUID);

    console.log('[BLE] Getting Data Characteristic...');
    const characteristic = await service.getCharacteristic(BLE_CONFIG.DATA_CHAR_UUID);
    bleState.dataChar = characteristic;

    // Start 100Hz notifications
    await characteristic.startNotifications();
    characteristic.addEventListener('characteristicvaluechanged', handleBLEPacketNotification);

    bleState.isConnected = true;
    if (bleState.isSimulating) stopSimulation();

    updateBleStatusUI('connected', `Streaming @ 100Hz from ${device.name || 'ESP32-REHAB'}`);
    console.log('[BLE] ✓ Successfully connected and streaming 100Hz synchronized dual-IMU packets!');

  } catch (err) {
    console.error('[BLE] Connection failed:', err);
    updateBleStatusUI('disconnected', `Error: ${err.message || 'Connection failed'}`);
    alert(`Bluetooth Connection Error: ${err.message}\nEnsure your ESP32 is powered on and within range, or click 'Simulate Real BLE Stream' to test models immediately.`);
  }
}

/**
 * Handle Web Bluetooth Disconnection Event
 */
function onBLEDisconnected() {
  console.log('[BLE] Device disconnected.');
  bleState.isConnected = false;
  bleState.dataChar = null;
  bleState.server = null;
  updateBleStatusUI('disconnected', 'Disconnected (Ready to Pair)');
}

/**
 * Disconnect Web Bluetooth
 */
function disconnectWebBluetooth() {
  if (bleState.device && bleState.device.gatt.connected) {
    bleState.device.gatt.disconnect();
  }
  onBLEDisconnected();
}

/**
 * Decode 52-Byte DualImuPacket from ESP32 BLE
 * struct __attribute__((packed)) DualImuPacket {
 *   uint32_t timestamp;  // 4 bytes: ESP32 millis()
 *   float s1_ax, s1_ay, s1_az, s1_gx, s1_gy, s1_gz; // 24 bytes Proximal IMU
 *   float s2_ax, s2_ay, s2_az, s2_gx, s2_gy, s2_gz; // 24 bytes Distal IMU
 * };
 */
function handleBLEPacketNotification(event) {
  const dv = event.target.value;
  if (!dv || dv.byteLength < 52) return;

  const sample = {
    timestamp: dv.getUint32(0, true),
    s1_ax: dv.getFloat32(4, true),
    s1_ay: dv.getFloat32(8, true),
    s1_az: dv.getFloat32(12, true),
    s1_gx: dv.getFloat32(16, true),
    s1_gy: dv.getFloat32(20, true),
    s1_gz: dv.getFloat32(24, true),
    s2_ax: dv.getFloat32(28, true),
    s2_ay: dv.getFloat32(32, true),
    s2_az: dv.getFloat32(36, true),
    s2_gx: dv.getFloat32(40, true),
    s2_gy: dv.getFloat32(44, true),
    s2_gz: dv.getFloat32(48, true)
  };

  processIncomingSample(sample);
}

/**
 * Toggle Realistic Hardware Simulation Mode
 */
function toggleHardwareSimulation() {
  if (bleState.isSimulating) {
    stopSimulation();
  } else {
    startSimulation();
  }
}

function startSimulation() {
  if (bleState.isConnected) disconnectWebBluetooth();

  bleState.isSimulating = true;
  bleState.simPhase = 0;
  const simPanel = document.getElementById('sim-controls-panel');
  if (simPanel) simPanel.style.display = 'block';

  const btnSim = document.getElementById('btn-ble-simulate');
  if (btnSim) {
    btnSim.classList.add('active');
    document.getElementById('sim-btn-text').textContent = 'Stop Simulation';
  }

  updateBleStatusUI('simulating', `Simulating Real 100Hz Stream (${bleState.simActivity})`);

  // Run simulation interval at 10ms (100Hz)
  const dt = 0.01; // 10ms
  bleState.simTimer = setInterval(() => {
    bleState.simPhase += dt;
    const sample = generateSimulatedSample(bleState.simActivity, bleState.simPhase);
    processIncomingSample(sample);
  }, 10);
}

function stopSimulation() {
  bleState.isSimulating = false;
  if (bleState.simTimer) clearInterval(bleState.simTimer);
  bleState.simTimer = null;

  const simPanel = document.getElementById('sim-controls-panel');
  if (simPanel) simPanel.style.display = 'none';

  const btnSim = document.getElementById('btn-ble-simulate');
  if (btnSim) {
    btnSim.classList.remove('active');
    document.getElementById('sim-btn-text').textContent = 'Simulate Real BLE Stream';
  }

  if (!bleState.isConnected) {
    updateBleStatusUI('disconnected', 'Disconnected (Ready to Pair)');
  }
}

/**
 * Generate 100Hz 12-channel kinematic sample matching clinical movement patterns
 */
function generateSimulatedSample(activity, t) {
  let s1_ax = 0.85, s1_ay = 0.0, s1_az = 0.2, s1_gx = 0.0, s1_gy = 0.0, s1_gz = 0.0;
  let s2_ax = 0.90, s2_ay = 0.0, s2_az = 0.3, s2_gx = 0.0, s2_gy = 0.0, s2_gz = 0.0;

  const act = activity.toLowerCase();
  if (act.includes('walk')) {
    const f = 1.4;
    s1_ax = 0.85 + 0.35 * Math.sin(2 * Math.PI * f * t) + 0.1 * Math.sin(4 * Math.PI * f * t);
    s1_ay = -0.05 + 0.15 * Math.cos(2 * Math.PI * f * t);
    s1_az = 0.25 + 0.25 * Math.sin(2 * Math.PI * f * t);
    s1_gx = -15.0 + 35.0 * Math.sin(2 * Math.PI * f * t);
    s1_gy = -25.0 + 110.0 * Math.cos(2 * Math.PI * f * t);
    s1_gz = 10.0 * Math.sin(2 * Math.PI * f * t);

    s2_ax = 0.95 + 0.55 * Math.sin(2 * Math.PI * f * t + 0.3);
    s2_ay = -0.08 + 0.18 * Math.cos(2 * Math.PI * f * t);
    s2_az = 0.35 + 0.40 * Math.sin(2 * Math.PI * f * t + 0.3);
    s2_gx = -20.0 + 45.0 * Math.sin(2 * Math.PI * f * t);
    s2_gy = -40.0 + 160.0 * Math.cos(2 * Math.PI * f * t);
    s2_gz = 15.0 * Math.sin(2 * Math.PI * f * t);

  } else if (act.includes('heel-toe')) {
    const f = 1.0;
    s1_ax = 0.70 + 0.45 * Math.sin(2 * Math.PI * f * t);
    s1_ay = -0.02 + 0.10 * Math.cos(2 * Math.PI * f * t);
    s1_az = 0.40 + 0.35 * Math.sin(2 * Math.PI * f * t);
    s1_gx = 25.0 * Math.sin(2 * Math.PI * f * t);
    s1_gy = 140.0 * Math.sin(2 * Math.PI * f * t);
    s1_gz = 5.0 * Math.cos(2 * Math.PI * f * t);

    s2_ax = 0.80 + 0.60 * Math.sin(2 * Math.PI * f * t + 0.2);
    s2_ay = -0.04 + 0.12 * Math.cos(2 * Math.PI * f * t);
    s2_az = 0.50 + 0.50 * Math.sin(2 * Math.PI * f * t + 0.2);
    s2_gx = 30.0 * Math.sin(2 * Math.PI * f * t);
    s2_gy = 190.0 * Math.sin(2 * Math.PI * f * t);
    s2_gz = 8.0 * Math.cos(2 * Math.PI * f * t);

  } else if (act.includes('squat')) {
    const f = 0.4;
    s1_ax = 0.50 + 0.60 * Math.sin(2 * Math.PI * f * t);
    s1_ay = 0.05 * Math.sin(2 * Math.PI * f * t);
    s1_az = 0.85 - 0.45 * Math.cos(2 * Math.PI * f * t);
    s1_gx = 5.0 * Math.sin(2 * Math.PI * f * t);
    s1_gy = 45.0 * Math.sin(2 * Math.PI * f * t);
    s1_gz = 4.0 * Math.sin(2 * Math.PI * f * t);

    s2_ax = 0.85 + 0.20 * Math.sin(2 * Math.PI * f * t);
    s2_ay = 0.02 * Math.sin(2 * Math.PI * f * t);
    s2_az = 0.45 - 0.25 * Math.cos(2 * Math.PI * f * t);
    s2_gx = 8.0 * Math.sin(2 * Math.PI * f * t);
    s2_gy = 65.0 * Math.sin(2 * Math.PI * f * t);
    s2_gz = 5.0 * Math.sin(2 * Math.PI * f * t);

  } else if (act.includes('knee')) {
    const f = 0.8;
    s1_ax = 0.35 + 0.50 * Math.sin(2 * Math.PI * f * t);
    s1_ay = -0.10 + 0.20 * Math.cos(2 * Math.PI * f * t);
    s1_az = 0.70 + 0.30 * Math.sin(2 * Math.PI * f * t);
    s1_gx = -40.0 + 80.0 * Math.sin(2 * Math.PI * f * t);
    s1_gy = -15.0 + 50.0 * Math.cos(2 * Math.PI * f * t);
    s1_gz = 20.0 * Math.sin(2 * Math.PI * f * t);

    s2_ax = 0.40 + 0.70 * Math.sin(2 * Math.PI * f * t);
    s2_ay = -0.15 + 0.25 * Math.cos(2 * Math.PI * f * t);
    s2_az = 0.80 + 0.40 * Math.sin(2 * Math.PI * f * t);
    s2_gx = -60.0 + 120.0 * Math.sin(2 * Math.PI * f * t);
    s2_gy = -20.0 + 75.0 * Math.cos(2 * Math.PI * f * t);
    s2_gz = 30.0 * Math.sin(2 * Math.PI * f * t);

  } else if (act.includes('hip')) {
    const f = 0.7;
    s1_ax = 0.60 + 0.25 * Math.sin(2 * Math.PI * f * t);
    s1_ay = 0.10 + 0.55 * Math.sin(2 * Math.PI * f * t);
    s1_az = 0.65 + 0.20 * Math.cos(2 * Math.PI * f * t);
    s1_gx = 45.0 * Math.sin(2 * Math.PI * f * t);
    s1_gy = 10.0 * Math.cos(2 * Math.PI * f * t);
    s1_gz = 35.0 * Math.sin(2 * Math.PI * f * t);

    s2_ax = 0.70 + 0.30 * Math.sin(2 * Math.PI * f * t);
    s2_ay = 0.15 + 0.70 * Math.sin(2 * Math.PI * f * t);
    s2_az = 0.70 + 0.25 * Math.cos(2 * Math.PI * f * t);
    s2_gx = 55.0 * Math.sin(2 * Math.PI * f * t);
    s2_gy = 15.0 * Math.cos(2 * Math.PI * f * t);
    s2_gz = 45.0 * Math.sin(2 * Math.PI * f * t);

  } else { // Abnormal gait / Tremor
    const f = 1.1;
    const noise = (Math.random() - 0.5) * 0.25;
    s1_ax = 0.75 + 0.30 * Math.sin(2 * Math.PI * f * t) + noise;
    s1_ay = -0.15 + 0.35 * Math.cos(2 * Math.PI * 1.8 * f * t) + noise;
    s1_az = 0.40 + 0.45 * Math.sin(2 * Math.PI * f * t) + noise;
    s1_gx = 30.0 * Math.sin(2 * Math.PI * f * t) + (Math.random() - 0.5) * 20;
    s1_gy = -45.0 + 85.0 * Math.cos(2 * Math.PI * f * t) + (Math.random() - 0.5) * 25;
    s1_gz = 25.0 * Math.sin(2 * Math.PI * f * t) + (Math.random() - 0.5) * 20;

    s2_ax = 0.80 + 0.40 * Math.sin(2 * Math.PI * f * t) + noise;
    s2_ay = -0.20 + 0.45 * Math.cos(2 * Math.PI * 1.8 * f * t) + noise;
    s2_az = 0.45 + 0.55 * Math.sin(2 * Math.PI * f * t) + noise;
    s2_gx = 40.0 * Math.sin(2 * Math.PI * f * t) + (Math.random() - 0.5) * 25;
    s2_gy = -60.0 + 110.0 * Math.cos(2 * Math.PI * f * t) + (Math.random() - 0.5) * 30;
    s2_gz = 35.0 * Math.sin(2 * Math.PI * f * t) + (Math.random() - 0.5) * 25;
  }

  // Slight sensor white noise
  const randNorm = () => (Math.random() - 0.5) * 0.02;
  return {
    timestamp: Date.now(),
    s1_ax: s1_ax + randNorm(),
    s1_ay: s1_ay + randNorm(),
    s1_az: s1_az + randNorm(),
    s1_gx: s1_gx + (Math.random() - 0.5) * 1.5,
    s1_gy: s1_gy + (Math.random() - 0.5) * 1.5,
    s1_gz: s1_gz + (Math.random() - 0.5) * 1.5,
    s2_ax: s2_ax + randNorm(),
    s2_ay: s2_ay + randNorm(),
    s2_az: s2_az + randNorm(),
    s2_gx: s2_gx + (Math.random() - 0.5) * 1.5,
    s2_gy: s2_gy + (Math.random() - 0.5) * 1.5,
    s2_gz: s2_gz + (Math.random() - 0.5) * 1.5
  };
}

/**
 * Main Sensor Processing Pipeline for Every Received Sample
 */
function processIncomingSample(sample) {
  bleState.packetCount++;
  bleState.rateCount++;

  // Update Sample Rate Hz Meter every 500ms
  const now = performance.now();
  if (now - bleState.lastRateCalcTime >= 500) {
    const elapsedSec = (now - bleState.lastRateCalcTime) / 1000.0;
    bleState.rateHz = (bleState.rateCount / elapsedSec).toFixed(1);
    bleState.rateCount = 0;
    bleState.lastRateCalcTime = now;

    const rateEl = document.getElementById('tel-rate');
    const packetEl = document.getElementById('tel-packets');
    const bufEl = document.getElementById('tel-buffer');
    if (rateEl) rateEl.textContent = `${bleState.rateHz} Hz`;
    if (packetEl) packetEl.textContent = bleState.packetCount.toLocaleString();
    if (bufEl) bufEl.textContent = `${bleState.rollingBuffer.length} / 200`;
  }

  // Calculate Ankle Joint Angle & Dynamic Range of Motion
  calculateAnkleKinematics(sample);

  // Buffer management
  bleState.rollingBuffer.push(sample);
  if (bleState.rollingBuffer.length > 200) {
    bleState.rollingBuffer.shift();
  }

  bleState.waveformBuffer.push(sample);
  if (bleState.waveformBuffer.length > 300) {
    bleState.waveformBuffer.shift();
  }

  // Record trial packet if active
  if (bleState.recording.isRecording) {
    bleState.recording.records.push({
      timestamp: sample.timestamp,
      ground_truth: bleState.activeGroundTruth,
      ...sample
    });
  }

  // Update scope numerical footer telemetry
  const ch = bleState.selectedChannel;
  const ax = sample[`${ch}_ax`];
  const ay = sample[`${ch}_ay`];
  const az = sample[`${ch}_az`];
  const gx = sample[`${ch}_gx`];
  const gy = sample[`${ch}_gy`];
  const gz = sample[`${ch}_gz`];

  const telAx = document.getElementById('tel-ax');
  const telAy = document.getElementById('tel-ay');
  const telAz = document.getElementById('tel-az');
  const telGx = document.getElementById('tel-gx');
  const telGy = document.getElementById('tel-gy');
  const telGz = document.getElementById('tel-gz');
  if (telAx) telAx.textContent = ax ? ax.toFixed(2) : '0.00';
  if (telAy) telAy.textContent = ay ? ay.toFixed(2) : '0.00';
  if (telAz) telAz.textContent = az ? az.toFixed(2) : '0.00';
  if (telGx) telGx.textContent = gx ? gx.toFixed(1) : '0.0';
  if (telGy) telGy.textContent = gy ? gy.toFixed(1) : '0.0';
  if (telGz) telGz.textContent = gz ? gz.toFixed(1) : '0.0';

  // Trigger Window Inference if buffer is full (200 samples) and interval reached
  if (bleState.rollingBuffer.length === 200 && !bleState.isInferencePending) {
    if (now - bleState.lastInferenceTime >= bleState.inferenceIntervalMs) {
      bleState.lastInferenceTime = now;
      dispatchWindowInference();
    }
  }
}

/**
 * Ankle Kinematics: Relative Joint Angle and Repetition Tracking
 */
function calculateAnkleKinematics(sample) {
  // Pitch angles for Sensor 1 (Tibia) and Sensor 2 (Foot)
  const s1_pitch = Math.atan2(-sample.s1_ax, Math.sqrt(sample.s1_ay * sample.s1_ay + sample.s1_az * sample.s1_az)) * (180.0 / Math.PI);
  const s2_pitch = Math.atan2(-sample.s2_ax, Math.sqrt(sample.s2_ay * sample.s2_ay + sample.s2_az * sample.s2_az)) * (180.0 / Math.PI);
  
  // Differential relative angle
  const relAngle = s2_pitch - s1_pitch;
  bleState.currentAnkleAngle = relAngle;

  if (relAngle > bleState.maxDorsiflexion) bleState.maxDorsiflexion = relAngle;
  if (relAngle < -bleState.maxPlantarflexion) bleState.maxPlantarflexion = Math.abs(relAngle);

  const totalArc = bleState.maxDorsiflexion + bleState.maxPlantarflexion;

  // Update ROM UI elements
  const angleEl = document.getElementById('rom-angle-val');
  const motionEl = document.getElementById('rom-motion-type');
  const maxDfEl = document.getElementById('rom-max-df');
  const maxPfEl = document.getElementById('rom-max-pf');
  const totalArcEl = document.getElementById('rom-total-arc');
  const dialArc = document.getElementById('rom-dial-arc');

  if (angleEl) angleEl.textContent = `${relAngle > 0 ? '+' : ''}${relAngle.toFixed(1)}°`;
  if (motionEl) {
    if (Math.abs(relAngle) < 3.0) motionEl.textContent = 'Neutral Stance (0°)';
    else if (relAngle > 0) motionEl.textContent = 'Dorsiflexion (Up)';
    else motionEl.textContent = 'Plantarflexion (Down)';
  }
  if (maxDfEl) maxDfEl.textContent = `${bleState.maxDorsiflexion.toFixed(1)}°`;
  if (maxPfEl) maxPfEl.textContent = `${bleState.maxPlantarflexion.toFixed(1)}°`;
  if (totalArcEl) totalArcEl.textContent = `${totalArc.toFixed(1)}°`;

  // Update Dial Arc SVG (Circumference 301.6)
  if (dialArc) {
    const clampedRatio = Math.max(-1, Math.min(1, relAngle / 45.0)); // -45 to +45 deg mapped to arc
    const offset = 150 - (clampedRatio * 100);
    dialArc.style.strokeDashoffset = offset;
  }

  // Repetition & Cadence Detection based on Gyroscope Y (Sagittal rotation)
  detectExerciseRepetition(sample.s1_gy);
}

/**
 * Repetition & Cadence Detection based on Gyroscope Peaks
 */
function detectExerciseRepetition(gy) {
  bleState.gyHistory.push(gy);
  if (bleState.gyHistory.length > 30) bleState.gyHistory.shift();

  // Peak detection threshold (gyro Y > 40 deg/s)
  const now = Date.now();
  if (Math.abs(gy) > 45.0 && (now - bleState.lastZeroCrossTime > 1200)) {
    bleState.repCount++;
    if (bleState.lastZeroCrossTime > 0) {
      const cycleSec = (now - bleState.lastZeroCrossTime) / 1000.0;
      bleState.repCadence = Math.round(60.0 / cycleSec);
    }
    bleState.lastZeroCrossTime = now;

    const repEl = document.getElementById('live-rep-count');
    const cadEl = document.getElementById('live-cadence');
    if (repEl) repEl.textContent = bleState.repCount;
    if (cadEl) cadEl.textContent = `${bleState.repCadence} reps/min`;
  }
}

/**
 * Real-Time 200-Sample Window Inference & Multi-Model Accuracy Checking
 */
async function dispatchWindowInference() {
  bleState.isInferencePending = true;

  try {
    const payload = {
      window: bleState.rollingBuffer,
      fs: BLE_CONFIG.SAMPLE_RATE_HZ,
      sensor_channel: bleState.selectedChannel,
      ground_truth: bleState.activeGroundTruth,
      apply_filter: true
    };

    const res = await fetch(`${API_BASE}/api/predict-raw-window`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!res.ok) throw new Error(`HTTP Error ${res.status}`);
    const data = await res.json();

    if (data.error) {
      console.warn('[Inference] Server returned error:', data.error);
      return;
    }

    renderRealTimeModelResults(data);

  } catch (err) {
    console.error('[Inference] Error during live model prediction:', err);
  } finally {
    bleState.isInferencePending = false;
  }
}

/**
 * Render All Models, Accuracy Verification, Stability, and Consensus Results
 */
function renderRealTimeModelResults(data) {
  const models = data.models || {};
  const consensus = data.consensus || {};
  const verification = data.verification || {};
  const stability = data.stability || {};
  const smoothness = data.smoothness || {};

  // 1. Update Session Accuracy Tracker
  bleState.sessionStats.totalWindows++;
  const isConsensusMatch = verification.consensus_matches;
  if (isConsensusMatch) {
    bleState.sessionStats.matchedWindows++;
  }

  // Update per-model hit counters
  Object.keys(bleState.sessionStats.perModelHits).forEach(mKey => {
    const modelMatched = verification.per_model_match ? verification.per_model_match[mKey] : false;
    bleState.sessionStats.perModelHits[mKey].total++;
    if (modelMatched) bleState.sessionStats.perModelHits[mKey].hits++;
  });

  const overallAcc = ((bleState.sessionStats.matchedWindows / bleState.sessionStats.totalWindows) * 100).toFixed(1);
  const accPctEl = document.getElementById('live-acc-percentage');
  const accTallyEl = document.getElementById('live-acc-tally');
  if (accPctEl) accPctEl.textContent = `${overallAcc}%`;
  if (accTallyEl) accTallyEl.textContent = `${bleState.sessionStats.matchedWindows} / ${bleState.sessionStats.totalWindows} Windows Matched`;

  // 2. Live Match Alert Banner
  const banner = document.getElementById('live-match-banner');
  const matchIcon = document.getElementById('live-match-icon');
  const matchStatus = document.getElementById('live-match-status');
  const matchReason = document.getElementById('live-match-reason');

  if (banner) {
    banner.classList.remove('is-match', 'is-mismatch');
    if (isConsensusMatch) {
      banner.classList.add('is-match');
      if (matchIcon) matchIcon.textContent = '✅';
      if (matchStatus) matchStatus.textContent = `Accurate Match: ${consensus.class}`;
      if (matchReason) matchReason.textContent = `All models recognized ${consensus.class} agreeing with ground-truth exercise.`;
    } else {
      banner.classList.add('is-mismatch');
      if (matchIcon) matchIcon.textContent = '⚠️';
      if (matchStatus) matchStatus.textContent = `Predicted: ${consensus.class} (Expected: ${verification.ground_truth})`;
      if (matchReason) matchReason.textContent = `Kinematic features indicated ${consensus.class} due to current acceleration/gyro harmonics.`;
    }
  }

  // 3. Model Consensus
  const consensusPctEl = document.getElementById('live-consensus-pct');
  const consensusDescEl = document.getElementById('live-consensus-desc');
  if (consensusPctEl) consensusPctEl.textContent = `${consensus.agreement_pct}%`;
  if (consensusDescEl) consensusDescEl.textContent = `${consensus.class} (${consensus.total_models} models evaluated)`;

  // 4. Render Model Cards
  renderModelCard('xgb', models['xgboost_opt'], verification.per_model_match ? verification.per_model_match['xgboost_opt'] : false, 'xgboost_opt');
  renderModelCard('rf', models['random_forest_opt'], verification.per_model_match ? verification.per_model_match['random_forest_opt'] : false, 'random_forest_opt');
  renderModelCard('tl', models['xgboost_transfer'], verification.per_model_match ? verification.per_model_match['xgboost_transfer'] : false, 'xgboost_transfer');

  // 5. Stability & Smoothness Card
  const stabScoreEl = document.getElementById('live-stab-score');
  const stabCatEl = document.getElementById('live-stab-category');
  const jerkEl = document.getElementById('live-jerk-val');
  const smoothEl = document.getElementById('live-smooth-status');
  const clinNotesEl = document.getElementById('live-clinical-notes');

  if (stabScoreEl) stabScoreEl.textContent = `${stability.score.toFixed(1)}%`;
  if (stabCatEl) stabCatEl.textContent = stability.category;
  if (jerkEl) jerkEl.textContent = `${smoothness.rms_jerk} m/s³`;
  if (smoothEl) smoothEl.textContent = smoothness.status;
  if (clinNotesEl) clinNotesEl.textContent = data.clinical_notes || data.recommendation;
}

/**
 * Render individual model evaluation card
 */
function renderModelCard(prefix, modelData, isMatch, modelKey) {
  if (!modelData) return;

  const iconEl = document.getElementById(`${prefix}-pred-icon`);
  const classEl = document.getElementById(`${prefix}-pred-class`);
  const confEl = document.getElementById(`${prefix}-pred-conf`);
  const matchEl = document.getElementById(`${prefix}-match-indicator`);
  const probasEl = document.getElementById(`${prefix}-probas-mini`);
  const sessionAccEl = document.getElementById(`${prefix}-session-acc`);

  const pClass = modelData.predicted_class || 'Unknown';
  if (iconEl) iconEl.textContent = EXERCISE_ICONS[pClass] || '⚡';
  if (classEl) classEl.textContent = pClass;
  if (confEl) confEl.textContent = `Confidence: ${modelData.confidence}%`;

  if (matchEl) {
    matchEl.className = `eval-match-indicator ${isMatch ? 'match' : 'mismatch'}`;
    matchEl.textContent = isMatch ? '✓ Matches Ground Truth' : 'Mismatch';
  }

  // Mini probability distribution bars
  if (probasEl && modelData.probabilities) {
    const sortedProbas = Object.entries(modelData.probabilities)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3); // Top 3

    probasEl.innerHTML = sortedProbas.map(([cName, pVal]) => `
      <div class="proba-mini-row">
        <span class="proba-mini-name">${cName}</span>
        <div class="proba-mini-bar-wrap">
          <div class="proba-mini-bar-fill" style="width:${pVal}%; background:${cName === pClass ? 'var(--cyan-primary)' : 'rgba(255,255,255,0.2)'};"></div>
        </div>
        <span class="proba-mini-val">${pVal}%</span>
      </div>
    `).join('');
  }

  // Model session accuracy
  if (sessionAccEl && bleState.sessionStats.perModelHits[modelKey]) {
    const stats = bleState.sessionStats.perModelHits[modelKey];
    const acc = stats.total > 0 ? ((stats.hits / stats.total) * 100).toFixed(1) : '--';
    sessionAccEl.textContent = `${acc}% (${stats.hits}/${stats.total})`;
  }
}

/**
 * Oscilloscope Canvas 60 FPS Engine
 */
function initOscilloscope() {
  const canvas = document.getElementById('canvas-oscilloscope');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  function resizeCanvas() {
    if (canvas.parentElement) {
      canvas.width = canvas.parentElement.clientWidth;
      canvas.height = canvas.parentElement.clientHeight || 220;
    }
  }
  window.addEventListener('resize', resizeCanvas);
  resizeCanvas();

  function drawScope() {
    requestAnimationFrame(drawScope);

    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    // Draw Dark Medical Oscilloscope Grid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
    ctx.lineWidth = 1;
    const stepX = 40;
    const stepY = 25;

    for (let x = 0; x < w; x += stepX) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
    }
    for (let y = 0; y < h; y += stepY) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }

    // Zero Midline
    const midY = h / 2;
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.2)';
    ctx.beginPath();
    ctx.moveTo(0, midY);
    ctx.lineTo(w, midY);
    ctx.stroke();

    const buffer = bleState.waveformBuffer;
    if (buffer.length < 2) return;

    const ch = bleState.selectedChannel;
    const chkAx = document.getElementById('chk-ax')?.checked ?? true;
    const chkAy = document.getElementById('chk-ay')?.checked ?? true;
    const chkAz = document.getElementById('chk-az')?.checked ?? true;
    const chkGx = document.getElementById('chk-gx')?.checked ?? true;
    const chkGy = document.getElementById('chk-gy')?.checked ?? true;
    const chkGz = document.getElementById('chk-gz')?.checked ?? true;

    const channels = [
      { key: `${ch}_ax`, color: '#f43f5e', scale: 35.0, enabled: chkAx }, // ~1g = 35px
      { key: `${ch}_ay`, color: '#10b981', scale: 35.0, enabled: chkAy },
      { key: `${ch}_az`, color: '#38bdf8', scale: 35.0, enabled: chkAz },
      { key: `${ch}_gx`, color: '#f59e0b', scale: 0.35, enabled: chkGx }, // ~100 deg/s = 35px
      { key: `${ch}_gy`, color: '#a855f7', scale: 0.35, enabled: chkGy },
      { key: `${ch}_gz`, color: '#06b6d4', scale: 0.35, enabled: chkGz }
    ];

    const dx = w / (buffer.length - 1);

    channels.forEach(channel => {
      if (!channel.enabled) return;

      ctx.strokeStyle = channel.color;
      ctx.lineWidth = 1.8;
      ctx.beginPath();

      for (let i = 0; i < buffer.length; i++) {
        const x = i * dx;
        const val = buffer[i][channel.key] || 0.0;
        const y = midY - (val * channel.scale);

        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    });
  }

  requestAnimationFrame(drawScope);
}

/**
 * Toggle Trial Session Recording
 */
function toggleTrialRecording() {
  const btnRec = document.getElementById('btn-toggle-record');
  const btnExport = document.getElementById('btn-export-csv');
  const recDot = document.getElementById('rec-dot');
  const recTitle = document.getElementById('rec-status-title');
  const recDesc = document.getElementById('rec-status-desc');
  const participantInput = document.getElementById('rec-participant-id');

  if (!bleState.recording.isRecording) {
    // Start Recording
    bleState.recording.isRecording = true;
    bleState.recording.records = [];
    bleState.recording.startTime = Date.now();
    bleState.recording.participantId = participantInput ? participantInput.value.trim() || 'P01' : 'P01';

    if (btnRec) {
      btnRec.classList.add('recording');
      document.getElementById('rec-btn-text').textContent = 'Stop Recording';
    }
    if (recDot) recDot.classList.add('active');
    if (recTitle) recTitle.textContent = `Recording Trial: ${bleState.recording.participantId} - ${bleState.activeGroundTruth}`;
    if (recDesc) recDesc.textContent = 'Buffering 100Hz 12-channel IMU samples with ground truth...';
    if (btnExport) btnExport.style.display = 'none';

  } else {
    // Stop Recording
    bleState.recording.isRecording = false;
    if (btnRec) {
      btnRec.classList.remove('recording');
      document.getElementById('rec-btn-text').textContent = 'Start Recording Trial';
    }
    if (recDot) recDot.classList.remove('active');
    if (recTitle) recTitle.textContent = `Recording Complete (${bleState.recording.records.length} Samples Captured)`;
    if (recDesc) recDesc.textContent = 'Trial recorded successfully. Click "Export CSV" to save data.';
    if (btnExport) btnExport.style.display = 'inline-flex';
  }
}

/**
 * Export Recorded Trial Data to CSV File
 */
async function exportSessionCSV() {
  if (bleState.recording.records.length === 0) {
    alert("No recorded samples to export. Start and stop a trial first.");
    return;
  }

  const pId = bleState.recording.participantId;
  const act = bleState.activeGroundTruth;

  // Attempt server-side save and download
  try {
    const res = await fetch(`${API_BASE}/api/export-session`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        participant_id: pId,
        component: 'ANKLE',
        activity: act,
        records: bleState.recording.records
      })
    });

    const data = await res.json();
    if (data.success) {
      alert(`Trial CSV successfully saved to server:\n${data.saved_path}\nTotal Samples: ${data.row_count}`);
    }
  } catch (err) {
    console.warn('[Export] Server export failed, downloading directly in browser...', err);
  }

  // Generate direct browser CSV download
  const headers = ['timestamp', 'ground_truth', 's1_ax', 's1_ay', 's1_az', 's1_gx', 's1_gy', 's1_gz', 's2_ax', 's2_ay', 's2_az', 's2_gx', 's2_gy', 's2_gz'];
  let csvContent = headers.join(',') + '\n';

  bleState.recording.records.forEach(row => {
    const line = headers.map(h => row[h] !== undefined ? row[h] : '').join(',');
    csvContent += line + '\n';
  });

  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.setAttribute('href', url);
  link.setAttribute('download', `${pId}_ANKLE_${act}_${Date.now()}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

/**
 * Helper to Update BLE Status Display
 */
function updateBleStatusUI(status, message) {
  const dot = document.getElementById('ble-connection-dot');
  const title = document.getElementById('ble-status-text');
  const btnConnect = document.getElementById('btn-ble-connect');
  const btnDisconnect = document.getElementById('btn-ble-disconnect');

  if (dot) {
    dot.className = 'ble-status-dot';
    if (status === 'connected') dot.classList.add('connected');
    else if (status === 'simulating') dot.classList.add('simulating');
  }

  if (title) title.textContent = message;

  if (btnConnect && btnDisconnect) {
    if (status === 'connected') {
      btnConnect.style.display = 'none';
      btnDisconnect.style.display = 'inline-flex';
    } else {
      btnConnect.style.display = 'inline-flex';
      btnDisconnect.style.display = 'none';
    }
  }
}
