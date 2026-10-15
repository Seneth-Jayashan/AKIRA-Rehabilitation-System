import demoData from '../data/demo_data.json';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { ShieldAlert, Database } from 'lucide-react';

export default function AnomalyDetection() {
  const data = demoData.predictions?.map((p, idx) => ({
    ...p,
    index: idx,
    fill: p.anomaly_status === 'ANOMALOUS' ? '#ef4444' : '#3b82f6'
  })) || [];

  const threshold = 0.003; 

  return (
    <div className="space-y-6">
      <div className="bg-slate-900 p-8 rounded-2xl border border-slate-800 shadow-xl">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-xl font-bold text-slate-100 flex items-center gap-3">
            <ShieldAlert className="text-cyan-400" />
            Machine Learning Diagnostics
          </h3>
          <div className="flex items-center gap-2 bg-slate-950 border border-slate-800 px-3 py-1.5 rounded-lg text-xs font-mono text-slate-400">
            <Database size={14} className="text-emerald-400" />
            08_RESULTS/predictions_demo.csv
          </div>
        </div>
        
        <p className="text-sm text-slate-400 mb-8 max-w-4xl border-l-2 border-slate-700 pl-4 py-1">
          Displays Autoencoder reconstruction errors (MSE). The model was trained offline. 
          Samples exceeding the 95th-percentile validation threshold are flagged. 
          <span className="text-slate-300 font-medium ml-1">Flagged samples represent algorithmic anomalies, not clinical diagnoses.</span>
        </p>
        
        {data.length > 0 ? (
          <div className="h-[400px] w-full bg-slate-950/50 p-6 rounded-2xl border border-slate-800/80 mb-8 relative">
            <div className="absolute top-4 right-4 bg-slate-900 border border-slate-700 px-3 py-1.5 rounded-lg text-xs shadow-lg z-10 flex gap-4">
              <div className="flex items-center gap-2"><span className="w-2.5 h-2.5 rounded-full bg-blue-500"></span> Normal</div>
              <div className="flex items-center gap-2"><span className="w-2.5 h-2.5 rounded-full bg-red-500"></span> Anomalous</div>
            </div>
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis type="number" dataKey="index" name="Sample" stroke="#475569" tick={{fill: '#64748b'}} />
                <YAxis type="number" dataKey="reconstruction_error" name="MSE Error" stroke="#475569" tick={{fill: '#64748b'}} />
                <Tooltip 
                  cursor={{ strokeDasharray: '3 3', stroke: '#475569' }} 
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', color: '#f1f5f9', borderRadius: '0.5rem' }}
                />
                <ReferenceLine y={threshold} stroke="#ef4444" strokeDasharray="3 3" label={{ position: 'top', value: 'Anomaly Threshold', fill: '#ef4444', fontSize: 12 }} />
                <Scatter name="Movements" data={data} fill="#8884d8" />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <div className="h-[400px] flex items-center justify-center text-slate-500 border border-dashed border-slate-800 rounded-xl bg-slate-950/50 mb-8">
            Prediction data unavailable.
          </div>
        )}

        <div className="bg-slate-950/50 rounded-2xl border border-slate-800/80 overflow-hidden shadow-inner">
          <div className="p-5 border-b border-slate-800/80 flex justify-between items-center bg-slate-900/50">
            <h3 className="text-sm font-bold text-slate-200 uppercase tracking-wider">Filtered Prediction Log</h3>
            <span className="text-xs text-slate-500 font-mono">Showing latest 10 samples</span>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm text-left text-slate-300">
              <thead className="bg-slate-900/80 text-slate-400 font-semibold uppercase text-xs tracking-wider">
                <tr>
                  <th className="px-6 py-4 border-b border-slate-800">Trial ID</th>
                  <th className="px-6 py-4 border-b border-slate-800">Participant</th>
                  <th className="px-6 py-4 border-b border-slate-800">Exercise</th>
                  <th className="px-6 py-4 border-b border-slate-800">MSE Error</th>
                  <th className="px-6 py-4 border-b border-slate-800">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/50">
                {data.slice(0, 10).map((row, i) => (
                  <tr key={i} className="hover:bg-slate-800/50 transition-colors">
                    <td className="px-6 py-4 font-mono text-xs text-blue-400">{row.trial_id || 'N/A'}</td>
                    <td className="px-6 py-4">{row.participant_id || 'Unknown'}</td>
                    <td className="px-6 py-4">{row.exercise_original || 'Unknown'}</td>
                    <td className="px-6 py-4 font-mono text-slate-400">{Number(row.reconstruction_error).toFixed(5)}</td>
                    <td className="px-6 py-4">
                      <span className={`px-3 py-1 rounded-full text-xs font-bold tracking-wide ${
                        row.anomaly_status === 'ANOMALOUS' 
                          ? 'bg-red-950 text-red-400 border border-red-900/50' 
                          : 'bg-emerald-950 text-emerald-400 border border-emerald-900/50'
                      }`}>
                        {row.anomaly_status}
                      </span>
                    </td>
                  </tr>
                ))}
                {data.length === 0 && (
                  <tr>
                    <td colSpan={5} className="px-6 py-8 text-center text-slate-500">No records found.</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
