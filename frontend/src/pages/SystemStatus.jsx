import { useState, useEffect } from 'react';
import axios from 'axios';
import { BarChart2, Info, AlertTriangle, FlaskConical } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

export default function SystemStatus({ mode }) {
  const [modelInfo, setModelInfo] = useState(null);
  
  useEffect(() => {
    if (mode === 'performance') {
      axios.get('http://localhost:8000/api/model-info').then(res => setModelInfo(res.data)).catch(console.error);
    }
  }, [mode]);

  if (mode === 'about') {
    return (
      <div className="space-y-6 max-w-4xl">
        <div className="flex items-center gap-3 border-b border-slate-800 pb-4 mb-6">
          <Info className="text-slate-400" size={28} />
          <h2 className="text-3xl font-black text-white tracking-tight">Reports & Limitations</h2>
        </div>
        
        <div className="bg-slate-900 p-8 rounded-3xl border border-slate-800 shadow-xl space-y-6">
          <h3 className="text-xl font-bold text-blue-400 flex items-center gap-3"><FlaskConical size={24}/> Research Prototypes</h3>
          <p className="text-slate-300 text-sm leading-relaxed">
            These models classify wrist movements using Quaternion-based spatial sequences or raw IMU anomalies. <strong>They are not clinically validated tools</strong> and should not be used to assess patient recovery after distal radius fracture (ORIF).
          </p>
          <div className="p-6 bg-amber-950/20 border border-amber-900/50 rounded-2xl text-amber-500 text-sm flex gap-4 items-start shadow-inner">
            <AlertTriangle size={24} className="shrink-0 mt-1" />
            <div className="space-y-2">
              <h4 className="font-bold text-amber-400 uppercase tracking-widest text-xs">Scientific Constraint</h4>
              <p className="leading-relaxed text-amber-500/90">These models predict the type of movement being executed or identify arbitrary variance. They cannot determine true relative wrist-angle kinematics or establish clinical readiness due to the lack of dual-IMU spatial synchronization.</p>
            </div>
          </div>
        </div>
      </div>
    )
  }

  // Performance Mode
  if (!modelInfo) return <div className="text-slate-500 p-8 text-center">Loading model metrics...</div>;

  const chartData = [
    { metric: 'Accuracy', RF: modelInfo.rf?.metrics?.accuracy * 100, CNN: modelInfo.cnn?.metrics?.accuracy * 100, XGB: modelInfo.xgb?.metrics?.accuracy * 100 },
    { metric: 'Macro F1', RF: modelInfo.rf?.metrics?.macro_f1 * 100, CNN: modelInfo.cnn?.metrics?.macro_f1 * 100, XGB: modelInfo.xgb?.metrics?.macro_f1 * 100 }
  ];

  return (
    <div className="space-y-6 max-w-5xl">
      <div className="flex items-center gap-3 border-b border-slate-800 pb-4 mb-6">
        <BarChart2 className="text-emerald-400" size={28} />
        <h2 className="text-3xl font-black text-white tracking-tight">Model Performance Comparison</h2>
      </div>

      <div className="bg-slate-900 p-8 rounded-3xl border border-slate-800 shadow-xl h-[450px]">
        <p className="text-slate-400 text-xs uppercase tracking-widest font-bold mb-6">Evaluated on Participant-Independent Test Split</p>
        <div className="h-[300px] w-full bg-slate-950/50 rounded-2xl p-6 border border-slate-800/80">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} layout="vertical" margin={{ left: 50, right: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis type="number" stroke="#475569" domain={[0, 100]} tickFormatter={(v) => `${v}%`} />
              <YAxis dataKey="metric" type="category" stroke="#94a3b8" />
              <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', borderRadius: '0.5rem', color: '#f8fafc' }} formatter={(val) => val.toFixed(1) + '%'} />
              <Legend wrapperStyle={{ paddingTop: '10px' }} />
              <Bar dataKey="RF" name="Random Forest (Model 2)" fill="#3b82f6" radius={[0, 4, 4, 0]} barSize={20} />
              <Bar dataKey="CNN" name="1D CNN (Model 3)" fill="#06b6d4" radius={[0, 4, 4, 0]} barSize={20} />
              <Bar dataKey="XGB" name="XGBoost (Model 4)" fill="#10b981" radius={[0, 4, 4, 0]} barSize={20} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
