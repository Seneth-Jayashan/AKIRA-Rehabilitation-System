import { useState, useEffect } from 'react';
import axios from 'axios';
import { Activity, Play, Pause, SkipForward, Cpu, ShieldCheck, AlertTriangle } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

export default function MovementAnalysis({ addHistory, mode }) {
  const [datasets, setDatasets] = useState([]);
  const [selectedFile, setSelectedFile] = useState('');
  const [selectedModel, setSelectedModel] = useState('all');
  const [sequenceData, setSequenceData] = useState([]);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [playing, setPlaying] = useState(false);
  const [idx, setIdx] = useState(0);

  useEffect(() => {
    if (mode === 'compare') {
      setSelectedModel('all');
    } else {
      setSelectedModel('rf');
    }
  }, [mode]);

  useEffect(() => {
    axios.get('http://localhost:8000/api/datasets').then(res => setDatasets(res.data)).catch(() => setError("Backend offline."));
  }, []);

  useEffect(() => {
    let interval;
    if (playing && sequenceData.length > 0 && idx < sequenceData.length - 1) {
      interval = setInterval(() => setIdx(p => p + 5), 50);
    } else if (idx >= sequenceData.length - 1) {
      setPlaying(false);
    }
    return () => clearInterval(interval);
  }, [playing, idx, sequenceData]);

  const loadSequence = (id) => {
    setSelectedFile(id); setPrediction(null); setSequenceData([]); setIdx(0); setPlaying(false);
    if (!id) return;
    axios.get(`http://localhost:8000/api/sequence/${id}`).then(res => setSequenceData(res.data.data)).catch(err => setError(err.response?.data?.detail));
  }
  
  const runPrediction = () => {
    setLoading(true); setError(null);
    axios.post('http://localhost:8000/api/predict', { sequence: sequenceData, model: selectedModel })
      .then(res => {
        setPrediction(res.data);
        if (addHistory) {
          if (selectedModel === 'all') {
            addHistory({ model: 'ALL', type: 'Comparison', result: 'Multi-Result', details: 'Executed 3 models concurrently' });
          } else {
            const data = res.data[selectedModel];
            if (data && !data.error) {
               addHistory({ model: selectedModel.toUpperCase(), type: 'Classification', result: data.prediction, details: `Conf: ${data.probabilities ? (Math.max(...Object.values(data.probabilities))*100).toFixed(1)+'%' : 'N/A'}` });
            }
          }
        }
      })
      .catch(err => setError(err.response?.data?.detail || "Prediction failed."))
      .finally(() => setLoading(false));
  }

  const chartData = sequenceData.slice(0, Math.max(1, idx)).map((s, i) => ({ time: i, W: s.w, X: s.x, Y: s.y, Z: s.z }));

  const renderResultCard = (data, title, color) => {
    if (!data) return null;
    if (data.error) return (
      <div className="bg-red-950/20 p-6 rounded-2xl border border-red-900/50 h-full flex flex-col justify-center min-h-[300px]">
         <AlertTriangle className="text-red-500 mb-4" size={32} />
         <h3 className="text-slate-200 font-bold mb-2">{title} Failed</h3>
         <p className="text-red-400 text-sm">{data.error}</p>
      </div>
    );
    const colorMap = {
      blue: { border: 'border-blue-900/50', bgGlow: 'bg-blue-600/10', textGlow: 'from-blue-400 to-blue-200', barBg: 'bg-blue-500' },
      cyan: { border: 'border-cyan-900/50', bgGlow: 'bg-cyan-600/10', textGlow: 'from-cyan-400 to-cyan-200', barBg: 'bg-cyan-500' },
      emerald: { border: 'border-emerald-900/50', bgGlow: 'bg-emerald-600/10', textGlow: 'from-emerald-400 to-emerald-200', barBg: 'bg-emerald-500' }
    };
    const colors = colorMap[color] || colorMap.blue;

    return (
      <div className={`bg-slate-900 p-8 rounded-3xl border ${colors.border} shadow-2xl relative overflow-hidden flex flex-col`}>
        <div className={`absolute top-0 right-0 w-32 h-32 rounded-full blur-[60px] -z-10 ${colors.bgGlow}`}></div>
        <h3 className="text-xs text-slate-400 font-bold uppercase tracking-widest mb-4">{title}</h3>
        <p className={`text-3xl font-black text-white bg-gradient-to-br ${colors.textGlow} bg-clip-text text-transparent mb-8 leading-tight`}>
          {data.prediction.replace('_', ' ')}
        </p>
        
        <div className="flex-1 space-y-4">
          {data.probabilities && Object.entries(data.probabilities).sort((a,b) => b[1] - a[1]).map(([cls, prob]) => (
            <div key={cls}>
              <div className="flex justify-between text-[10px] uppercase font-bold tracking-wider mb-1.5">
                <span className="text-slate-400">{cls.replace('_', ' ')}</span>
                <span className="text-white font-mono">{(prob * 100).toFixed(1)}%</span>
              </div>
              <div className="w-full bg-slate-950 rounded-full h-1.5 border border-slate-800 overflow-hidden">
                <div className={`${colors.barBg} h-full`} style={{ width: `${prob * 100}%` }}></div>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-6">
      <div className="bg-slate-900 p-8 rounded-2xl border border-slate-800 shadow-xl">
        <div className="flex justify-between items-center mb-6 border-b border-slate-800 pb-4">
          <h3 className="text-xl font-bold text-slate-100 flex items-center gap-3">
            <Cpu className="text-blue-400" /> Multi-Model Classifiers
          </h3>
          <span className="flex items-center gap-2 bg-emerald-950/30 text-emerald-500 px-3 py-1.5 rounded-lg text-xs font-semibold uppercase tracking-wide border border-emerald-900/50">
            <ShieldCheck size={14} /> LIVE API ACTIVE
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
          <div>
            <label className="block text-xs font-bold text-slate-400 uppercase tracking-widest mb-2">Select Quaternion Sequence</label>
            <select className="w-full bg-slate-950 border border-slate-700 rounded-lg px-4 py-3 text-slate-200 outline-none focus:border-blue-500" value={selectedFile} onChange={(e) => loadSequence(e.target.value)}>
              <option value="">-- Choose sequence --</option>
              {datasets.map(d => <option key={d.id} value={d.id}>{d.filename} ({d.participant})</option>)}
            </select>
          </div>
          <div>
            <label className="block text-xs font-bold text-slate-400 uppercase tracking-widest mb-2">Select Target Classifier</label>
            <select disabled={mode === 'compare'} className="w-full bg-slate-950 border border-slate-700 rounded-lg px-4 py-3 text-slate-200 outline-none focus:border-blue-500 disabled:opacity-50" value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)}>
              <option value="all">Compare All (RF, CNN, XGB)</option>
              <option value="rf">Model 2: Random Forest</option>
              <option value="cnn">Model 3: 1D CNN</option>
              <option value="xgb">Model 4: XGBoost</option>
            </select>
          </div>
        </div>

        {error && <div className="text-red-400 text-sm bg-red-900/20 p-4 rounded-xl border border-red-900 mb-6 flex items-center gap-2"><AlertTriangle size={18}/> {error}</div>}

        {sequenceData.length > 0 && (
          <div className="space-y-8">
            <div className="bg-slate-950/50 p-6 rounded-2xl border border-slate-800/80 h-[450px] flex flex-col">
               <div className="flex justify-between items-center mb-4">
                 <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest">Sequence Playback Visualization</h3>
                 <div className="flex gap-2">
                   <button onClick={() => setIdx(0)} className="p-2 bg-slate-800 rounded hover:bg-slate-700 text-slate-300"><SkipForward size={16} className="rotate-180"/></button>
                   <button onClick={() => setPlaying(!playing)} className="p-2 bg-blue-600 rounded hover:bg-blue-500 text-white shadow-lg shadow-blue-900/50">{playing ? <Pause size={16}/> : <Play size={16}/>}</button>
                 </div>
               </div>
               <div className="flex-1">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                      <XAxis dataKey="time" stroke="#475569" type="number" domain={[0, sequenceData.length]} />
                      <YAxis stroke="#475569" domain={[-1, 1]} />
                      <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b' }} />
                      <Legend />
                      <Line type="monotone" dataKey="W" stroke="#3b82f6" dot={false} strokeWidth={2} isAnimationActive={false} />
                      <Line type="monotone" dataKey="X" stroke="#ef4444" dot={false} strokeWidth={2} isAnimationActive={false} />
                      <Line type="monotone" dataKey="Y" stroke="#22c55e" dot={false} strokeWidth={2} isAnimationActive={false} />
                      <Line type="monotone" dataKey="Z" stroke="#eab308" dot={false} strokeWidth={2} isAnimationActive={false} />
                    </LineChart>
                  </ResponsiveContainer>
               </div>
            </div>

            <button onClick={runPrediction} disabled={loading} className="w-full py-5 bg-gradient-to-r from-blue-600 to-blue-500 hover:from-blue-500 hover:to-blue-400 disabled:opacity-50 text-white font-black tracking-wide rounded-2xl shadow-lg shadow-blue-900/20 transition-all flex items-center justify-center gap-3 text-lg">
              {loading ? 'Processing Multi-Model Pipelines...' : <><Play fill="currentColor" size={20}/> Execute Inference</>}
            </button>

            {prediction && (
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 pt-6 border-t border-slate-800">
                {selectedModel === 'all' || selectedModel === 'rf' ? renderResultCard(prediction.rf, 'Random Forest', 'blue') : null}
                {selectedModel === 'all' || selectedModel === 'cnn' ? renderResultCard(prediction.cnn, '1D CNN', 'cyan') : null}
                {selectedModel === 'all' || selectedModel === 'xgb' ? renderResultCard(prediction.xgb, 'XGBoost', 'emerald') : null}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
