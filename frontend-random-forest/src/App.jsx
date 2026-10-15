import { useState, useEffect } from 'react';
import axios from 'axios';
import { 
    LayoutDashboard, MonitorPlay, Activity, FileText, Info, AlertTriangle, Zap, FlaskConical, Play, ShieldAlert
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, BarChart, Bar } from 'recharts';

const API_URL = 'http://localhost:8000/api';

export default function App() {
  const [activeTab, setActiveTab] = useState('overview');
  const [sysStatus, setSysStatus] = useState(null);
  
  useEffect(() => {
    axios.get(`${API_URL}/health`).then(res => setSysStatus(res.data)).catch(() => setSysStatus({status: 'offline'}));
  }, []);

  const navItems = [
    { id: 'overview', icon: LayoutDashboard, label: 'Overview' },
    { id: 'prediction', icon: MonitorPlay, label: 'Movement Classifiers' },
    { id: 'anomaly', icon: ShieldAlert, label: 'Anomaly Detection' },
    { id: 'performance', icon: Activity, label: 'Model Performance' },
    { id: 'about', icon: Info, label: 'About the Models' },
  ];

  return (
    <div className="flex h-screen bg-slate-950 text-slate-100 font-sans overflow-hidden">
      {/* Sidebar */}
      <div className="w-64 bg-slate-900 border-r border-slate-800 flex flex-col shrink-0 z-20">
        <div className="p-6 border-b border-slate-800">
          <div className="flex items-center gap-2 mb-1">
            <Zap className="text-cyan-400" size={24} />
            <h1 className="text-lg font-bold bg-gradient-to-r from-blue-400 to-cyan-300 bg-clip-text text-transparent">
              AKIRA
            </h1>
          </div>
          <p className="text-[10px] uppercase tracking-widest text-slate-400 font-semibold mb-3">
            Wrist Movement Intelligence
          </p>
          <div className="px-2 py-1 bg-blue-950/50 border border-blue-900/50 rounded flex items-center gap-2">
            <FlaskConical size={12} className="text-blue-400" />
            <span className="text-[10px] text-blue-300 font-medium">Multi-Model Prototype</span>
          </div>
        </div>
        
        <nav className="flex-1 py-4 px-3 space-y-1 overflow-y-auto">
          {navItems.map(item => (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                activeTab === item.id 
                  ? 'bg-blue-600/10 text-blue-400 border border-blue-900/50' 
                  : 'text-slate-400 hover:text-slate-100 hover:bg-slate-800/50 border border-transparent'
              }`}
            >
              <item.icon size={18} />
              {item.label}
            </button>
          ))}
        </nav>
        
        <div className="p-4 border-t border-slate-800 text-xs">
          <div className="flex items-center gap-2 text-slate-400 mb-2">
            <span className={`w-2 h-2 rounded-full ${sysStatus?.status === 'ok' ? 'bg-emerald-500' : 'bg-red-500'}`}></span>
            Backend: {sysStatus?.status === 'ok' ? 'Online' : 'Offline'}
          </div>
          <div className="flex items-center gap-2 text-slate-400 mb-1">
             <span className={`w-2 h-2 rounded-full ${sysStatus?.rf_loaded ? 'bg-emerald-500' : 'bg-red-500'}`}></span>
            RF: {sysStatus?.rf_loaded ? 'Loaded' : 'Missing'}
          </div>
          <div className="flex items-center gap-2 text-slate-400 mb-1">
             <span className={`w-2 h-2 rounded-full ${sysStatus?.cnn_loaded ? 'bg-emerald-500' : 'bg-red-500'}`}></span>
            CNN: {sysStatus?.cnn_loaded ? 'Loaded' : 'Missing'}
          </div>
          <div className="flex items-center gap-2 text-slate-400 mb-1">
             <span className={`w-2 h-2 rounded-full ${sysStatus?.xgb_loaded ? 'bg-emerald-500' : 'bg-red-500'}`}></span>
            XGBoost: {sysStatus?.xgb_loaded ? 'Loaded' : 'Missing'}
          </div>
          <div className="flex items-center gap-2 text-slate-400">
             <span className={`w-2 h-2 rounded-full ${sysStatus?.ae_loaded ? 'bg-emerald-500' : 'bg-red-500'}`}></span>
            Autoencoder: {sysStatus?.ae_loaded ? 'Loaded' : 'Missing'}
          </div>
        </div>
      </div>

      <main className="flex-1 overflow-y-auto p-8 relative">
        {activeTab === 'overview' && <OverviewPage />}
        {activeTab === 'prediction' && <PredictionPage />}
        {activeTab === 'anomaly' && <AnomalyPage />}
        {activeTab === 'performance' && <PerformancePage />}
        {activeTab === 'about' && <AboutPage />}
      </main>
    </div>
  );
}

function OverviewPage() {
  const [modelInfo, setModelInfo] = useState(null);
  
  useEffect(() => {
    axios.get(`${API_URL}/model-info`).then(res => setModelInfo(res.data)).catch(console.error);
  }, []);

  return (
    <div className="max-w-6xl space-y-6">
      <h2 className="text-2xl font-bold">Multi-Model Overview Dashboard</h2>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* RF Info */}
        <div className="bg-slate-900 p-6 rounded-2xl border border-slate-800">
          <h3 className="text-sm text-slate-400 font-medium uppercase tracking-wider mb-4">Model 2: Random Forest</h3>
          <div className="space-y-4">
            <div className="flex justify-between border-b border-slate-800 pb-2">
              <span className="text-slate-300">Input Type</span>
              <span className="font-semibold text-white">20 Temporal Stats</span>
            </div>
            <div className="flex justify-between border-b border-slate-800 pb-2">
              <span className="text-slate-300">Test Accuracy</span>
              <span className="font-semibold text-blue-400">{modelInfo?.rf?.metrics ? (modelInfo.rf.metrics.accuracy * 100).toFixed(1) + '%' : 'N/A'}</span>
            </div>
          </div>
        </div>
        
        {/* CNN Info */}
        <div className="bg-slate-900 p-6 rounded-2xl border border-slate-800">
          <h3 className="text-sm text-slate-400 font-medium uppercase tracking-wider mb-4">Model 3: 1D CNN</h3>
          <div className="space-y-4">
            <div className="flex justify-between border-b border-slate-800 pb-2">
              <span className="text-slate-300">Input Type</span>
              <span className="font-semibold text-white">Sequence Tensor (250x4)</span>
            </div>
            <div className="flex justify-between border-b border-slate-800 pb-2">
              <span className="text-slate-300">Test Accuracy</span>
              <span className="font-semibold text-cyan-400">{modelInfo?.cnn?.metrics ? (modelInfo.cnn.metrics.accuracy * 100).toFixed(1) + '%' : 'N/A'}</span>
            </div>
          </div>
        </div>
        
        {/* XGB Info */}
        <div className="bg-slate-900 p-6 rounded-2xl border border-slate-800">
          <h3 className="text-sm text-slate-400 font-medium uppercase tracking-wider mb-4">Model 4: XGBoost</h3>
          <div className="space-y-4">
            <div className="flex justify-between border-b border-slate-800 pb-2">
              <span className="text-slate-300">Input Type</span>
              <span className="font-semibold text-white">20 Temporal Stats</span>
            </div>
            <div className="flex justify-between border-b border-slate-800 pb-2">
              <span className="text-slate-300">Test Accuracy</span>
              <span className="font-semibold text-emerald-400">{modelInfo?.xgb?.metrics ? (modelInfo.xgb.metrics.accuracy * 100).toFixed(1) + '%' : 'N/A'}</span>
            </div>
          </div>
        </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-slate-900 p-6 rounded-2xl border border-slate-800">
          <h3 className="text-sm text-slate-400 font-medium uppercase tracking-wider mb-4">Supported Movement Classes (Models 2,3,4)</h3>
          <div className="grid grid-cols-2 gap-3">
             {['NEUTRAL', 'RADIAL_DEVIATION', 'WRIST_FLEXION', 'ULNAR_DEVIATION'].map(c => (
               <div key={c} className="bg-slate-950 p-3 rounded-lg border border-slate-800 flex items-center justify-center text-center text-sm font-medium text-slate-300">
                 {c.replace('_', ' ')}
               </div>
             ))}
          </div>
        </div>
        
        <div className="bg-slate-900 p-6 rounded-2xl border border-slate-800">
          <h3 className="text-sm text-slate-400 font-medium uppercase tracking-wider mb-4">Model 1: Autoencoder (Anomaly Detection)</h3>
          <p className="text-slate-400 text-sm mb-4">
            The Autoencoder processes raw accelerometer and gyroscope data (unlike the quaternion classifiers) to identify anomalous movement execution based on a strict 95th-percentile validation MSE threshold.
          </p>
          <div className="flex justify-between border-b border-slate-800 pb-2">
            <span className="text-slate-300">Threshold Used</span>
            <span className="font-semibold text-amber-400">{modelInfo?.ae?.threshold ? modelInfo.ae.threshold.toFixed(5) : 'N/A'}</span>
          </div>
        </div>
      </div>
    </div>
  )
}

function AnomalyPage() {
  const [datasets, setDatasets] = useState([]);
  const [selectedFile, setSelectedFile] = useState('');
  const [sequenceData, setSequenceData] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    axios.get(`${API_URL}/ae-datasets`).then(res => setDatasets(res.data)).catch(err => setError("Could not load AE datasets."));
  }, []);

  const loadSequence = (id) => {
    setSelectedFile(id);
    setPrediction(null);
    if (!id) {
        setSequenceData(null);
        return;
    }
    const match = datasets.find(d => d.id === id);
    if (match) setSequenceData(match.data);
  }
  
  const runPrediction = () => {
    setLoading(true);
    setError(null);
    axios.post(`${API_URL}/predict-anomaly`, { features: sequenceData })
      .then(res => setPrediction(res.data))
      .catch(err => setError(err.response?.data?.detail || "Prediction failed."))
      .finally(() => setLoading(false));
  }

  return (
    <div className="max-w-4xl space-y-6">
      <h2 className="text-2xl font-bold">Autoencoder Anomaly Detection</h2>
      
      <div className="bg-slate-900 p-6 rounded-2xl border border-slate-800 shadow-xl">
        <label className="block text-sm text-slate-400 font-medium mb-2">Select Pre-processed Kinematic Sample (Raw IMU features)</label>
        <select 
          className="w-full bg-slate-950 border border-slate-800 rounded-lg px-4 py-3 text-slate-200 outline-none focus:border-blue-500 transition-colors"
          value={selectedFile}
          onChange={(e) => loadSequence(e.target.value)}
        >
          <option value="">-- Choose sample --</option>
          {datasets.map(d => (
            <option key={d.id} value={d.id}>{d.description}</option>
          ))}
        </select>
      </div>

      {error && <p className="text-red-400 mt-2 text-sm bg-red-900/20 p-4 rounded-xl border border-red-900">{error}</p>}

      {sequenceData && (
        <div className="space-y-6">
          <button 
            onClick={runPrediction}
            disabled={loading}
            className="w-full py-4 bg-amber-600 hover:bg-amber-500 disabled:opacity-50 text-white font-bold rounded-xl shadow-lg transition-colors flex items-center justify-center gap-3 text-lg"
          >
            {loading ? 'Processing...' : <><ShieldAlert /> Evaluate Reconstruction Error</>}
          </button>
          
          {prediction && !prediction.error && (
             <div className={`p-6 rounded-2xl border shadow-xl ${prediction.is_anomaly ? 'bg-red-900/20 border-red-900/50' : 'bg-emerald-900/20 border-emerald-900/50'}`}>
               <h3 className="text-sm text-slate-400 font-medium uppercase tracking-wider mb-2">Autoencoder Output</h3>
               <div className="flex justify-between items-center mb-6">
                 <p className={`text-3xl font-black ${prediction.is_anomaly ? 'text-red-400' : 'text-emerald-400'}`}>
                   {prediction.is_anomaly ? 'ANOMALOUS (Irregular Movement)' : 'NORMAL (Standard Movement)'}
                 </p>
               </div>
               <div className="space-y-2 text-sm">
                 <div className="flex justify-between border-b border-slate-800/50 pb-2">
                   <span className="text-slate-400">Mean Squared Error (MSE)</span>
                   <span className="text-white font-mono">{prediction.mse_score.toFixed(6)}</span>
                 </div>
                 <div className="flex justify-between border-b border-slate-800/50 pb-2">
                   <span className="text-slate-400">Validated Threshold (95th Percentile)</span>
                   <span className="text-white font-mono">{prediction.threshold_used.toFixed(6)}</span>
                 </div>
               </div>
             </div>
          )}
        </div>
      )}
    </div>
  )
}

function PredictionPage() {
  const [datasets, setDatasets] = useState([]);
  const [selectedFile, setSelectedFile] = useState('');
  const [selectedModel, setSelectedModel] = useState('all');
  const [sequenceData, setSequenceData] = useState([]);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    axios.get(`${API_URL}/datasets`).then(res => setDatasets(res.data)).catch(err => setError("Could not load datasets. Backend offline."));
  }, []);

  const loadSequence = (id) => {
    setSelectedFile(id);
    setPrediction(null);
    setSequenceData([]);
    if (!id) return;
    axios.get(`${API_URL}/sequence/${id}`).then(res => setSequenceData(res.data.data)).catch(err => setError(err.response?.data?.detail || "Failed to load sequence."));
  }
  
  const runPrediction = () => {
    setLoading(true);
    setError(null);
    axios.post(`${API_URL}/predict`, { sequence: sequenceData, model: selectedModel })
      .then(res => setPrediction(res.data))
      .catch(err => setError(err.response?.data?.detail || "Prediction failed."))
      .finally(() => setLoading(false));
  }

  const chartData = sequenceData.map((s, i) => ({ time: i, W: s.w, X: s.x, Y: s.y, Z: s.z }));

  const renderCard = (modelKey, title, accentColor) => {
    if (!prediction || !prediction[modelKey] || prediction[modelKey].error) return null;
    const data = prediction[modelKey];
    
    return (
      <div className={`bg-slate-900 p-6 rounded-2xl border border-${accentColor}-900/50 shadow-xl shadow-${accentColor}-900/20`}>
        <h3 className="text-sm text-slate-400 font-medium uppercase tracking-wider mb-2">{title}</h3>
        <p className={`text-2xl font-black text-white bg-gradient-to-r from-${accentColor}-400 to-${accentColor}-200 bg-clip-text text-transparent mb-6`}>
          {data.prediction.replace('_', ' ')}
        </p>
        {data.probabilities && (
          <div className="space-y-3">
            {Object.entries(data.probabilities).sort((a,b) => b[1] - a[1]).map(([cls, prob]) => (
              <div key={cls}>
                <div className="flex justify-between text-[10px] mb-1">
                  <span className="text-slate-300">{cls}</span>
                  <span className="text-white font-mono">{(prob * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full bg-slate-950 rounded-full h-1 border border-slate-800">
                  <div className={`bg-${accentColor}-500 h-1 rounded-full`} style={{ width: `${prob * 100}%` }}></div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="max-w-6xl space-y-6">
      <h2 className="text-2xl font-bold">Movement Classifiers Workspace</h2>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-slate-900 p-6 rounded-2xl border border-slate-800 shadow-xl">
          <label className="block text-sm text-slate-400 font-medium mb-2">Select Quaternion Sequence</label>
          <select 
            className="w-full bg-slate-950 border border-slate-800 rounded-lg px-4 py-3 text-slate-200 outline-none focus:border-blue-500 transition-colors"
            value={selectedFile}
            onChange={(e) => loadSequence(e.target.value)}
          >
            <option value="">-- Choose sequence --</option>
            {datasets.map(d => (
              <option key={d.id} value={d.id}>{d.filename} ({d.participant})</option>
            ))}
          </select>
        </div>
        
        <div className="bg-slate-900 p-6 rounded-2xl border border-slate-800 shadow-xl">
          <label className="block text-sm text-slate-400 font-medium mb-2">Select Classifier to Run</label>
          <select 
            className="w-full bg-slate-950 border border-slate-800 rounded-lg px-4 py-3 text-slate-200 outline-none focus:border-blue-500 transition-colors"
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
          >
            <option value="all">Compare All (RF, CNN, XGB)</option>
            <option value="rf">Model 2: Random Forest</option>
            <option value="cnn">Model 3: 1D CNN</option>
            <option value="xgb">Model 4: XGBoost</option>
          </select>
        </div>
      </div>
      
      {error && <p className="text-red-400 mt-2 text-sm bg-red-900/20 p-4 rounded-xl border border-red-900">{error}</p>}

      {sequenceData.length > 0 && (
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
          <div className="xl:col-span-2 space-y-6">
            <div className="bg-slate-900 p-6 rounded-2xl border border-slate-800 shadow-xl h-[400px]">
              <h3 className="text-sm text-slate-400 font-medium uppercase tracking-wider mb-4">Input Data: Quaternion Sequence</h3>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="time" stroke="#475569" />
                  <YAxis stroke="#475569" domain={[-1, 1]} />
                  <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', color: '#f8fafc' }} />
                  <Legend />
                  <Line type="monotone" dataKey="W" stroke="#3b82f6" dot={false} strokeWidth={2} />
                  <Line type="monotone" dataKey="X" stroke="#ef4444" dot={false} strokeWidth={2} />
                  <Line type="monotone" dataKey="Y" stroke="#22c55e" dot={false} strokeWidth={2} />
                  <Line type="monotone" dataKey="Z" stroke="#eab308" dot={false} strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
            
            <button 
              onClick={runPrediction}
              disabled={loading}
              className="w-full py-4 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white font-bold rounded-xl shadow-lg transition-colors flex items-center justify-center gap-3 text-lg"
            >
              {loading ? 'Processing...' : <><Play /> Execute Inference</>}
            </button>
          </div>

          <div className="xl:col-span-1 space-y-6">
             {!prediction && (
               <div className="bg-slate-900/50 p-6 rounded-2xl border border-slate-800/50 h-full flex flex-col items-center justify-center text-slate-500 text-center">
                 <FlaskConical size={48} className="mb-4 opacity-50" />
                 <p>Run prediction to view classification comparisons.</p>
               </div>
             )}
             
             {renderCard('rf', 'Random Forest', 'blue')}
             {renderCard('cnn', '1D CNN', 'cyan')}
             {renderCard('xgb', 'XGBoost', 'emerald')}
          </div>
        </div>
      )}
    </div>
  )
}

function PerformancePage() {
  const [modelInfo, setModelInfo] = useState(null);
  useEffect(() => {
    axios.get(`${API_URL}/model-info`).then(res => setModelInfo(res.data)).catch(console.error);
  }, []);

  if (!modelInfo) return <div className="p-8 text-center text-slate-400">Loading metrics...</div>

  const chartData = [
    {
      metric: 'Accuracy',
      RF: modelInfo.rf?.metrics ? modelInfo.rf.metrics.accuracy * 100 : 0,
      CNN: modelInfo.cnn?.metrics ? modelInfo.cnn.metrics.accuracy * 100 : 0,
      XGB: modelInfo.xgb?.metrics ? modelInfo.xgb.metrics.accuracy * 100 : 0,
    },
    {
      metric: 'Macro F1',
      RF: modelInfo.rf?.metrics ? modelInfo.rf.metrics.macro_f1 * 100 : 0,
      CNN: modelInfo.cnn?.metrics ? modelInfo.cnn.metrics.macro_f1 * 100 : 0,
      XGB: modelInfo.xgb?.metrics ? modelInfo.xgb.metrics.macro_f1 * 100 : 0,
    }
  ];

  return (
    <div className="max-w-5xl space-y-6">
      <h2 className="text-2xl font-bold">Model Performance Comparison</h2>
      <p className="text-slate-400 text-sm">Classifiers evaluated exclusively on the same 5 unseen held-out participants.</p>
      
      <div className="bg-slate-900 p-6 rounded-xl border border-slate-800 shadow-xl h-[300px]">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} layout="vertical" margin={{ left: 50, right: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis type="number" stroke="#475569" domain={[0, 100]} tickFormatter={(v) => `${v}%`} />
            <YAxis dataKey="metric" type="category" stroke="#94a3b8" />
            <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', color: '#f8fafc' }} formatter={(val) => val.toFixed(1) + '%'} />
            <Legend />
            <Bar dataKey="RF" name="Random Forest" fill="#3b82f6" radius={[0, 4, 4, 0]} />
            <Bar dataKey="CNN" name="1D CNN" fill="#06b6d4" radius={[0, 4, 4, 0]} />
            <Bar dataKey="XGB" name="XGBoost" fill="#10b981" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-slate-900 p-6 rounded-xl border border-slate-800">
           <h3 className="font-bold text-slate-100 mb-2">Random Forest Stats</h3>
           <p className="text-sm text-slate-400">Available at <code className="bg-slate-950 p-1 border border-slate-800 text-[10px]">08_RESULTS/random_forest_quaternion_classification_report.csv</code></p>
        </div>
        <div className="bg-slate-900 p-6 rounded-xl border border-slate-800">
           <h3 className="font-bold text-slate-100 mb-2">1D CNN Stats</h3>
           <p className="text-sm text-slate-400">Available at <code className="bg-slate-950 p-1 border border-slate-800 text-[10px]">08_RESULTS/1d_cnn_quaternion_classification_report.csv</code></p>
        </div>
        <div className="bg-slate-900 p-6 rounded-xl border border-slate-800">
           <h3 className="font-bold text-slate-100 mb-2">XGBoost Stats</h3>
           <p className="text-sm text-slate-400">Available at <code className="bg-slate-950 p-1 border border-slate-800 text-[10px]">08_RESULTS/xgboost_quaternion_classification_report.csv</code></p>
        </div>
      </div>
    </div>
  )
}

function AboutPage() {
  return (
    <div className="max-w-3xl space-y-6">
      <h2 className="text-2xl font-bold mb-6">About the Models</h2>
      
      <div className="bg-slate-900 p-6 rounded-2xl border border-slate-800 space-y-4">
        <h3 className="text-lg font-bold text-blue-400 flex items-center gap-2"><FlaskConical size={20}/> Research Prototypes</h3>
        <p className="text-slate-300 text-sm leading-relaxed">
          These models classify wrist movements using Quaternion-based spatial sequences or raw IMU anomalies. **They are not clinically validated tools** and should not be used to assess patient recovery after distal radius fracture (ORIF).
        </p>
        <div className="p-4 bg-amber-950/30 border border-amber-900/50 rounded-xl text-amber-500 text-sm flex gap-3">
          <AlertTriangle size={20} className="shrink-0" />
          <p>These models predict the type of movement being executed or identify arbitrary variance. They cannot determine true relative wrist-angle kinematics or establish clinical readiness.</p>
        </div>
      </div>
    </div>
  )
}
