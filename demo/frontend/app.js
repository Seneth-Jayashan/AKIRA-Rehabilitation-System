const classMapping = {
    "ElbowFlexCali": { human: "Calibration: Elbow Flexion/Ext", category: "Calibration Only" },
    "ElbowProCali": { human: "Calibration: Elbow Pro/Sup", category: "Calibration Only" },
    "N cali": { human: "Calibration: N-Pose", category: "Calibration Only" },
    "ScapulaElevationCali": { human: "Calibration: Scapula Elevation", category: "Calibration Only" },
    "ScapulaProtractionCali": { human: "Calibration: Scapula Protraction", category: "Calibration Only" },
    "ShoAbdCali": { human: "Calibration: Shoulder Abduction", category: "Calibration Only" },
    "ShoFlexCali": { human: "Calibration: Shoulder Flexion", category: "Calibration Only" },
    "ShoIntRotCali": { human: "Calibration: Shoulder Rotation", category: "Calibration Only" },
    "WristDevCali": { human: "Calibration: Wrist Deviation", category: "Calibration Only" },
    "WristFlexCali": { human: "Calibration: Wrist Flexion/Ext", category: "Calibration Only" },
    "task 1": { human: "Dynamic Elbow Flexion/Extension", category: "Direct Elbow-Relevant Movement" },
    "task 2": { human: "Dynamic Forearm Pronation/Supination", category: "Direct Elbow-Relevant Movement" },
    "task 3": { human: "Shoulder Flexion", category: "Supporting Upper-Limb Movement" },
    "task 4": { human: "Shoulder Abduction", category: "Supporting Upper-Limb Movement" },
    "task 5": { human: "Shoulder Rotation", category: "Supporting Upper-Limb Movement" },
    "task 6": { human: "Wrist Flexion/Extension", category: "Supporting Upper-Limb Movement" },
    "task 7": { human: "Wrist Deviation", category: "Supporting Upper-Limb Movement" },
    "task 8": { human: "Functional: Drinking From a Cup", category: "Functional Activity" },
    "task 9": { human: "Functional: Box Off Shelf", category: "Functional Activity" },
    "task 10": { human: "Functional: Drawing Circles", category: "Functional Activity" }
};

const infoDefinitions = {
    'task1': {
        movement: 'Dynamic Elbow Flexion/Extension',
        relevance: 'Primary elbow/forearm target',
        dataset: 'task 1',
        purpose: 'Evaluate elbow kinematic estimation and supporting movement classification.'
    },
    'task2': {
        movement: 'Dynamic Forearm Pronation/Supination',
        relevance: 'Primary elbow/forearm target',
        dataset: 'task 2',
        purpose: 'Evaluate forearm rotational kinematic estimation and supporting movement classification.'
    },
    'task3': { movement: 'Dynamic Shoulder Flexion', relevance: 'Supporting upper-limb movement', dataset: 'task 3', purpose: 'Evaluate supporting movement classification.' },
    'task4': { movement: 'Dynamic Shoulder Abduction', relevance: 'Supporting upper-limb movement', dataset: 'task 4', purpose: 'Evaluate supporting movement classification.' },
    'task5': { movement: 'Dynamic Shoulder Rotation', relevance: 'Supporting upper-limb movement', dataset: 'task 5', purpose: 'Evaluate supporting movement classification.' },
    'task6': { movement: 'Dynamic Wrist Flexion/Extension', relevance: 'Supporting upper-limb movement', dataset: 'task 6', purpose: 'Evaluate supporting movement classification.' },
    'task7': { movement: 'Dynamic Wrist Deviation', relevance: 'Supporting upper-limb movement', dataset: 'task 7', purpose: 'Evaluate supporting movement classification.' },
    'task8': { movement: 'Drinking From a Cup', relevance: 'Functional activity', dataset: 'task 8', purpose: 'Evaluate movement classification during a functional upper-limb activity.' },
    'task9': { movement: 'Box Off Shelf', relevance: 'Functional activity', dataset: 'task 9', purpose: 'Evaluate movement classification during a functional upper-limb activity.' },
    'task10': { movement: 'Drawing Circles', relevance: 'Functional activity', dataset: 'task 10', purpose: 'Evaluate movement classification during a functional upper-limb activity.' },
    'cali_elbowflex': { movement: 'Elbow Flexion/Extension Calibration', relevance: 'Calibration only', dataset: 'ElbowFlexCali', purpose: 'Dataset calibration protocol.' },
    'cali_elbowpro': { movement: 'Forearm Pronation/Supination Calibration', relevance: 'Calibration only', dataset: 'ElbowProCali', purpose: 'Dataset calibration protocol.' },
    'cali_npose': { movement: 'N-Pose Calibration', relevance: 'Calibration only', dataset: 'N cali', purpose: 'Dataset calibration protocol.' },
    'cali_scapelev': { movement: 'Scapula Elevation Calibration', relevance: 'Calibration only', dataset: 'ScapulaElevationCali', purpose: 'Dataset calibration protocol.' },
    'cali_scappro': { movement: 'Scapula Protraction Calibration', relevance: 'Calibration only', dataset: 'ScapulaProtractionCali', purpose: 'Dataset calibration protocol.' },
    'cali_shoabd': { movement: 'Shoulder Abduction Calibration', relevance: 'Calibration only', dataset: 'ShoAbdCali', purpose: 'Dataset calibration protocol.' },
    'cali_shoflex': { movement: 'Shoulder Flexion Calibration', relevance: 'Calibration only', dataset: 'ShoFlexCali', purpose: 'Dataset calibration protocol.' },
    'cali_shorot': { movement: 'Shoulder Internal/External Rotation Calibration', relevance: 'Calibration only', dataset: 'ShoIntRotCali', purpose: 'Dataset calibration protocol.' },
    'cali_wristdev': { movement: 'Wrist Deviation Calibration', relevance: 'Calibration only', dataset: 'WristDevCali', purpose: 'Dataset calibration protocol.' },
    'cali_wristflex': { movement: 'Wrist Flexion/Extension Calibration', relevance: 'Calibration only', dataset: 'WristFlexCali', purpose: 'Dataset calibration protocol.' },
    'error': { movement: 'Classification Error', relevance: 'Model Limitation', dataset: 'Classification Error Example', purpose: 'An example where the model\'s movement classification differs from the ground-truth class.' }
};

let elbowChart, forearmChart, probChart;
let replayInterval = null;
let currentEndIdx = 240;
let maxIdx = 240;
let isPlaying = false;
let currentExample = "";

async function checkIntegrity() {
    try {
        const res = await fetch('/api/model-info');
        const data = await res.json();
        
        document.getElementById('sha-hash').innerText = data.expected_hash;
        
        if (!data.integrity_passed) {
            document.getElementById('integrity-warning').classList.remove('hidden');
            document.getElementById('hash-box').classList.add('hidden');
        }
    } catch (e) {
        console.error("Failed to check integrity", e);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    checkIntegrity();
    initCharts();
});

document.getElementById('demo-select').addEventListener('change', (e) => {
    const key = e.target.value;
    const info = infoDefinitions[key];
    document.getElementById('desc-movement').innerText = info.movement;
    document.getElementById('desc-relevance').innerText = info.relevance;
    document.getElementById('desc-dataset').innerText = info.dataset;
    document.getElementById('desc-purpose').innerText = info.purpose;
    resetReplay();
});

document.getElementById('btn-play').addEventListener('click', () => {
    if (isPlaying) return;
    startReplay();
});

document.getElementById('btn-pause').addEventListener('click', () => {
    pauseReplay();
});

document.getElementById('btn-reset').addEventListener('click', () => {
    resetReplay();
});

document.getElementById('speed-select').addEventListener('change', () => {
    if (isPlaying) {
        pauseReplay();
        startReplay();
    }
});

async function resetReplay() {
    pauseReplay();
    currentEndIdx = 240;
    currentExample = document.getElementById('demo-select').value;
    document.getElementById('s-state').innerText = "READY";
    document.getElementById('s-state').style.color = "blue";
    
    // Fetch max length
    const res = await fetch(`/api/demo-length?example_id=${currentExample}`);
    const data = await res.json();
    maxIdx = data.length;
    updateProgress();
    
    document.getElementById('vis-section').style.opacity = '1';
    document.getElementById('error-msg').innerText = "";
    
    // Do one initial render
    await fetchAndRender();
}

function startReplay() {
    if (currentEndIdx >= maxIdx) {
        currentEndIdx = 240; // loop back
    }
    
    currentExample = document.getElementById('demo-select').value;
    isPlaying = true;
    document.getElementById('s-state').innerText = "LIVE REPLAY";
    document.getElementById('s-state').style.color = "green";
    
    const speed = parseFloat(document.getElementById('speed-select').value);
    
    // 120Hz = 120 samples per 1000ms.
    // If we step every 100ms, that's 12 samples.
    // So 12 * speed.
    const intervalMs = 100;
    const step = Math.round(12 * speed);
    
    replayInterval = setInterval(async () => {
        currentEndIdx += step;
        if (currentEndIdx >= maxIdx) {
            currentEndIdx = maxIdx;
            pauseReplay();
            document.getElementById('s-state').innerText = "FINISHED";
            document.getElementById('s-state').style.color = "gray";
        }
        await fetchAndRender();
        updateProgress();
    }, intervalMs);
}

function pauseReplay() {
    isPlaying = false;
    if (replayInterval) clearInterval(replayInterval);
    if (document.getElementById('s-state').innerText === "LIVE REPLAY") {
        document.getElementById('s-state').innerText = "PAUSED";
        document.getElementById('s-state').style.color = "orange";
    }
}

function updateProgress() {
    document.getElementById('s-window').innerText = `${currentEndIdx} / ${maxIdx}`;
    document.getElementById('replay-progress').value = (currentEndIdx / maxIdx) * 100;
}

async function fetchAndRender() {
    try {
        const res = await fetch(`/api/predict-stream?example_id=${currentExample}&end_idx=${currentEndIdx}`);
        if (!res.ok) throw new Error("Stream inference failed");
        
        const data = await res.json();
        renderVisualization(data);
    } catch (e) {
        pauseReplay();
        document.getElementById('error-msg').innerText = e.message;
    }
}

function initCharts() {
    const labels = Array.from({length: 240}, (_, i) => (i/120).toFixed(2));
    
    elbowChart = new Chart(document.getElementById('elbowChart'), {
        type: 'line',
        data: { labels, datasets: [] },
        options: { maintainAspectRatio: false, animation: false, scales: { y: { suggestedMin: 0, suggestedMax: 150 } } }
    });
    
    forearmChart = new Chart(document.getElementById('forearmChart'), {
        type: 'line',
        data: { labels, datasets: [] },
        options: { maintainAspectRatio: false, animation: false, scales: { y: { suggestedMin: -180, suggestedMax: 180 } } }
    });
    
    probChart = new Chart(document.getElementById('probChart'), {
        type: 'bar',
        data: { labels: [], datasets: [] },
        options: { maintainAspectRatio: false, indexAxis: 'y', animation: false, scales: { x: { max: 1 } } }
    });

    // Baseline Comparison Chart
    const baselineCtx = document.getElementById('baselineChart');
    if (baselineCtx) {
        new Chart(baselineCtx, {
            type: 'bar',
            data: {
                labels: ['TCN', 'LSTM', 'Linear Baseline'],
                datasets: [
                    {
                        label: 'Elbow MAE',
                        data: [13.38, 17.05, 46.15],
                        backgroundColor: 'rgba(220, 53, 69, 0.7)'
                    },
                    {
                        label: 'Forearm MAE',
                        data: [24.41, 27.65, 25.11],
                        backgroundColor: 'rgba(0, 86, 179, 0.7)'
                    }
                ]
            },
            options: {
                maintainAspectRatio: false,
                animation: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Model Comparison — MAE'
                    }
                },
                scales: {
                    y: {
                        title: {
                            display: true,
                            text: 'Mean Absolute Error (degrees)'
                        },
                        beginAtZero: true
                    }
                }
            }
        });
    }
}

function renderVisualization(data) {
    const predRaw = data.exercise_prediction;
    const predInfo = classMapping[predRaw] || { human: "Unknown", category: "Unknown" };
    
    document.getElementById('pred-class-raw').innerText = predRaw;
    document.getElementById('pred-class-human').innerText = predInfo.human;
    document.getElementById('pred-category').innerText = predInfo.category;
    
    const probsArray = Object.values(data.class_probabilities);
    const maxProb = Math.max(...probsArray);
    document.getElementById('pred-conf').innerText = (maxProb * 100).toFixed(1);
    
    if (data.gt_exercise) {
        const gtRaw = data.gt_exercise;
        const gtInfo = classMapping[gtRaw] || { human: "Unknown" };
        document.getElementById('gt-class-raw').innerText = gtRaw;
        document.getElementById('gt-class-human').innerText = gtInfo.human;
    } else {
        document.getElementById('gt-class-raw').innerText = "N/A";
        document.getElementById('gt-class-human').innerText = "N/A";
    }
    
    const currentE = data.elbow_prediction[data.elbow_prediction.length - 1];
    const currentF = data.forearm_prediction[data.forearm_prediction.length - 1];
    
    document.getElementById('m-e-cur').innerText = currentE.toFixed(2);
    document.getElementById('m-f-cur').innerText = currentF.toFixed(2);
    
    document.getElementById('m-e-rom').innerText = data.rom_elbow.toFixed(2);
    document.getElementById('m-f-rom').innerText = data.rom_forearm.toFixed(2);
    document.getElementById('m-reps').innerText = data.repetition_count;
    document.getElementById('m-vel').innerText = data.velocity_elbow.toFixed(2);
    
    const labels = Array.from({length: 240}, (_, i) => ((currentEndIdx - 240 + i) / 120).toFixed(2));
    
    const e_datasets = [{
        label: 'Predicted Elbow Angle (°)',
        data: data.elbow_prediction,
        borderColor: 'red',
        fill: false,
        borderDash: [5, 5],
        pointRadius: 0
    }];
    if (data.gt_elbow) {
        e_datasets.unshift({
            label: 'Ground Truth Elbow Angle (°)',
            data: data.gt_elbow,
            borderColor: 'green',
            fill: false,
            pointRadius: 0
        });
    }
    elbowChart.data.labels = labels;
    elbowChart.data.datasets = e_datasets;
    elbowChart.update();
    
    const f_datasets = [{
        label: 'Predicted Forearm Angle (°)',
        data: data.forearm_prediction,
        borderColor: 'orange',
        fill: false,
        borderDash: [5, 5],
        pointRadius: 0
    }];
    if (data.gt_forearm) {
        f_datasets.unshift({
            label: 'Ground Truth Forearm Angle (°)',
            data: data.gt_forearm,
            borderColor: 'blue',
            fill: false,
            pointRadius: 0
        });
    }
    forearmChart.data.labels = labels;
    forearmChart.data.datasets = f_datasets;
    forearmChart.update();
    
    const rawClasses = Object.keys(data.class_probabilities);
    const humanClasses = rawClasses.map(rc => classMapping[rc] ? classMapping[rc].human : rc);
    
    probChart.data.labels = humanClasses;
    probChart.data.datasets = [{
        label: 'Class Probability',
        data: probsArray,
        backgroundColor: 'rgba(0, 86, 179, 0.5)'
    }];
    probChart.update();
}
