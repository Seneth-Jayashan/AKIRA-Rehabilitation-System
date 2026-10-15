import { useState, useEffect } from 'react';
import axios from 'axios';
import demoData from '../data/demo_data.json';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, Cell } from 'recharts';
import { ShieldAlert, Database, Play, AlertTriangle } from 'lucide-react';

export default function AnomalyDetection({ addHistory }) {
  const [datasets, setDatasets] = useState([]);
  const [selectedFile, setSelectedFile] = useState('');
  const [sequenceData, setSequenceData] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [liveHistory, setLiveHistory] = useState([]);

  useEffect(() => {
    axios.get('http://localhost:8000/api/ae-datasets').then(res => setDatasets(res.data)).catch(() => setError("Backend offline."));
  }, []);

  const loadSequence = (id) => {
    setSelectedFile(id); setPrediction(null);
    if (!id) { setSequenceData(null); return; }
    const match = datasets.find(d => d.id === id);
    if (match) setSequenceData(match.data);
  }
  
  const runPrediction = () => {
    setLoading(true); setError(null);
    axios.post('http://localhost:8000/api/predict-anomaly', { features: sequenceData })
      .then(res => {
        setPrediction(res.data);
        if (addHistory) {
          addHistory({ model: 'Autoencoder', type: 'Anomaly Detection', result: res.data.is_anomaly ? 'ANOMALOUS' : 'NORMAL', details: `MSE: ${res.data.mse_score.toFixed(6)}` });
        }
        setLiveHistory(prev => [...prev.slice(-19), { 
          id: selectedFile, 
          index: demoData.predictions.length + prev.length,
          reconstruction_error: res.data.mse_score, 
          anomaly_status: res.data.is_anomaly ? 'ANOMALOUS' : 'NORMAL'
        }]);
      })
      .catch(err => setError(err.response?.data?.detail || "Prediction failed."))
      .finally(() => setLoading(false));
  }

  const baseData = demoData.predictions?.map((p, idx) => ({ ...p, index: idx })) || [];
  const combinedData = [...baseData, ...liveHistory].map(p => ({
    ...p, fill: p.anomaly_status === 'ANOMALOUS' ? '#ef4444' : '#3b82f6'
  }));

  const threshold = prediction?.threshold_used || 0.0128194;

  return (
    <div className="space-y-6">
      <div className="bg-slate-900 p-8 rounded-2xl border border-slate-800 shadow-xl">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-xl font-bold text-slate-100 flex items-center gap-3">
            <ShieldAlert className="text-cyan-400" /> Autoencoder Machine Learning Diagnostics
          </h3>
          <div className="flex items-center gap-2 bg-slate-950 border border-slate-800 px-3 py-1.5 rounded-lg text-xs font-mono text-slate-400">
            <Database size={14} className="text-emerald-400" /> live-inference
          </div>
        </div>

        <p className="text-sm text-slate-400 mb-8 max-w-4xl border-l-2 border-slate-700 pl-4 py-1">
          Displays Autoencoder reconstruction errors (MSE). The model dynamically loads the 95th-percentile validation threshold. 
          <span className="text-slate-300 font-medium ml-1">Flagged samples represent algorithmic anomalies, not clinical diagnoses.</span>
        </p>

        {/* Live Evaluation Box */}
        <div className="bg-slate-950/50 p-6 rounded-2xl border border-slate-800/80 mb-8 grid grid-cols-1 lg:grid-cols-2 gap-6">
           <div className="space-y-4">
              <label className="block text-xs font-bold text-slate-400 uppercase tracking-widest">Select Kinematic Sample (Raw IMU features)</label>
              <select className="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-3 text-slate-200 outline-none focus:border-cyan-500" value={selectedFile} onChange={(e) => loadSequence(e.target.value)}>
                <option value="">-- Choose sample sequence --</option>
                {datasets.map(d => <option key={d.id} value={d.id}>{d.description}</option>)}
              </select>
              {sequenceData && (
                <button onClick={runPrediction} disabled={loading} className="w-full py-4 bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 text-white font-bold rounded-lg shadow-lg transition-colors flex items-center justify-center gap-2">
                  {loading ? 'Evaluating...' : <><Play size={18} fill="currentColor"/> Compute Reconstruction Error</>}
                </button>
              )}
              {error && <div className="text-red-400 text-sm bg-red-900/20 p-3 rounded-lg border border-red-900 flex items-center gap-2"><AlertTriangle size={16}/> {error}</div>}
           </div>
           
           {prediction && (
             <div className={`p-6 rounded-xl border shadow-xl flex flex-col justify-center ${prediction.is_anomaly ? 'bg-red-950/40 border-red-900/50' : 'bg-emerald-950/40 border-emerald-900/50'}`}>
                <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-4">Live Prediction Result</h3>
                <p className={`text-4xl font-black mb-6 ${prediction.is_anomaly ? 'text-red-400' : 'text-emerald-400'}`}>{prediction.is_anomaly ? 'ANOMALOUS' : 'NORMAL'}</p>
                <div className="flex justify-between border-b border-slate-800/50 pb-2 mb-2 text-sm">
                  <span className="text-slate-400">Mean Squared Error</span>
                  <span className="text-white font-mono">{prediction.mse_score.toFixed(6)}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-slate-400">Validated Threshold</span>
                  <span className="text-amber-400 font-mono">{prediction.threshold_used.toFixed(6)}</span>
                </div>
             </div>
           )}
        </div>

        {/* Scatter Plot */}
        <div className="h-[400px] w-full bg-slate-950/50 p-6 rounded-2xl border border-slate-800/80 mb-8 relative">
          <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-2">Historical Reconstruction Errors</h3>
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis type="number" dataKey="index" name="Sample" stroke="#475569" tick={{fill: '#64748b'}} />
              <YAxis type="number" dataKey="reconstruction_error" name="MSE Error" stroke="#475569" domain={[0, 'auto']} tick={{fill: '#64748b'}} />
              <Tooltip cursor={{ strokeDasharray: '3 3', stroke: '#475569' }} contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', color: '#f1f5f9' }} />
              <ReferenceLine y={threshold} stroke="#f59e0b" strokeDasharray="3 3" label={{ position: 'top', value: 'Anomaly Threshold', fill: '#f59e0b', fontSize: 12 }} />
              <Scatter name="Movements" data={combinedData}>
                {combinedData.map((entry, i) => (
                  <Cell key={`cell-${i}`} fill={entry.fill} />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
