import { History, Activity } from 'lucide-react';

export default function SessionHistory({ history = [] }) {
  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3 border-b border-slate-800 pb-4 mb-6">
        <History className="text-purple-400" size={28} />
        <h2 className="text-3xl font-black text-white tracking-tight">Prediction History</h2>
      </div>

      <div className="bg-slate-900 p-8 rounded-2xl border border-slate-800 shadow-xl">
        {history.length === 0 ? (
          <div className="h-[300px] flex flex-col items-center justify-center text-slate-500 border border-dashed border-slate-800 rounded-xl bg-slate-950/50">
            <Activity size={48} className="mb-4 opacity-20" />
            <p>No predictions have been run in this active session.</p>
          </div>
        ) : (
          <div className="space-y-3">
            {history.map(h => (
              <div key={h.id} className="bg-slate-950 p-4 rounded-xl border border-slate-800/80 flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div className="flex items-center gap-4">
                  <span className="text-xs font-mono text-slate-500 bg-slate-900 px-2 py-1 rounded">{h.timestamp}</span>
                  <span className="text-sm font-bold text-slate-300 w-32">{h.model}</span>
                  <span className="text-xs text-slate-400 uppercase tracking-wider">{h.type}</span>
                </div>
                <div className="flex items-center gap-4 md:text-right">
                   <span className={`font-black text-lg ${h.result === 'ANOMALOUS' ? 'text-red-400' : h.result === 'NORMAL' ? 'text-emerald-400' : 'text-blue-400'}`}>
                     {h.result.replace('_', ' ')}
                   </span>
                   <span className="text-xs text-slate-500 font-mono w-32 bg-slate-900 px-2 py-1 rounded">{h.details}</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
