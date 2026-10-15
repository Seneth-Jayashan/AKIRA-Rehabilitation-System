import { LineChart, Activity, Database } from 'lucide-react';

export default function DataExplorer() {
  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3 border-b border-slate-800 pb-4 mb-6">
        <LineChart className="text-cyan-400" size={28} />
        <h2 className="text-3xl font-black text-white tracking-tight">Sensor Data Explorer</h2>
      </div>

      <div className="bg-slate-900 p-8 rounded-3xl border border-slate-800 flex items-center gap-6 text-slate-400 shadow-xl max-w-4xl">
        <div className="p-4 bg-slate-950 rounded-2xl border border-slate-800 shadow-inner">
          <Database size={48} className="text-cyan-500" />
        </div>
        <div>
          <h3 className="text-xl font-bold text-slate-200 mb-2">Full Dataset Processing</h3>
          <p className="text-sm leading-relaxed">
            This section is reserved for interactive exploration of the raw generic IMU sequences and full quaternion batches. The required backend endpoints for full dataset pagination and timeline rendering are not currently active in the API Gateway.
          </p>
        </div>
      </div>
    </div>
  )
}
