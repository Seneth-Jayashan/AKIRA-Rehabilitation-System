import demoData from '../data/demo_data.json';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Activity, FileText } from 'lucide-react';

export default function DataExplorer() {
  const tsData = demoData.timeseries || [];

  return (
    <div className="space-y-6">
      <div className="bg-slate-900 p-8 rounded-2xl border border-slate-800 shadow-xl">
        <div className="flex justify-between items-center mb-8 border-b border-slate-800 pb-4">
          <h3 className="text-xl font-bold text-slate-100 flex items-center gap-3">
            <Activity className="text-cyan-400" />
            Raw IMU Telemetry
          </h3>
          <div className="flex items-center gap-2 bg-slate-950 border border-slate-800 px-3 py-1.5 rounded-lg text-xs font-mono text-slate-400 shadow-inner">
            <FileText size={14} className="text-blue-400" />
            WatchHand_standard.csv (Sample)
          </div>
        </div>
        
        {tsData.length > 0 ? (
          <div className="space-y-10">
            {/* Accelerometer */}
            <div className="bg-slate-950/50 p-6 rounded-2xl border border-slate-800/80">
              <div className="flex items-center gap-3 mb-6">
                <div className="h-5 w-1.5 bg-blue-500 rounded-full shadow-[0_0_10px_rgba(59,130,246,0.5)]"></div>
                <h4 className="text-sm font-bold text-slate-200 uppercase tracking-widest">Accelerometer (g)</h4>
              </div>
              <div className="h-[320px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={tsData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#1e293b" />
                    <XAxis dataKey="timestamp" hide />
                    <YAxis stroke="#475569" tick={{fill: '#64748b', fontSize: 12}} />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', color: '#f1f5f9', borderRadius: '0.5rem' }}
                      itemStyle={{ fontSize: 13, fontWeight: 500 }}
                    />
                    <Legend iconType="circle" wrapperStyle={{ paddingTop: '20px', fontSize: 13 }} />
                    <Line type="monotone" dataKey="ax" stroke="#ef4444" strokeWidth={2.5} dot={false} name="Accel X" isAnimationActive={false} />
                    <Line type="monotone" dataKey="ay" stroke="#22c55e" strokeWidth={2.5} dot={false} name="Accel Y" isAnimationActive={false} />
                    <Line type="monotone" dataKey="az" stroke="#3b82f6" strokeWidth={2.5} dot={false} name="Accel Z" isAnimationActive={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
            
            {/* Gyroscope */}
            <div className="bg-slate-950/50 p-6 rounded-2xl border border-slate-800/80">
              <div className="flex items-center gap-3 mb-6">
                <div className="h-5 w-1.5 bg-purple-500 rounded-full shadow-[0_0_10px_rgba(168,85,247,0.5)]"></div>
                <h4 className="text-sm font-bold text-slate-200 uppercase tracking-widest">Gyroscope (rad/s)</h4>
              </div>
              <div className="h-[320px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={tsData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#1e293b" />
                    <XAxis dataKey="timestamp" hide />
                    <YAxis stroke="#475569" tick={{fill: '#64748b', fontSize: 12}} />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', color: '#f1f5f9', borderRadius: '0.5rem' }}
                      itemStyle={{ fontSize: 13, fontWeight: 500 }}
                    />
                    <Legend iconType="circle" wrapperStyle={{ paddingTop: '20px', fontSize: 13 }} />
                    <Line type="monotone" dataKey="gx" stroke="#ef4444" strokeWidth={2.5} dot={false} name="Gyro X" isAnimationActive={false} />
                    <Line type="monotone" dataKey="gy" stroke="#22c55e" strokeWidth={2.5} dot={false} name="Gyro Y" isAnimationActive={false} />
                    <Line type="monotone" dataKey="gz" stroke="#3b82f6" strokeWidth={2.5} dot={false} name="Gyro Z" isAnimationActive={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        ) : (
          <div className="py-24 flex flex-col items-center justify-center text-slate-500 bg-slate-950/50 rounded-2xl border border-slate-800 border-dashed">
            <Activity size={48} className="text-slate-700 mb-4" />
            <p className="text-lg">No demonstration telemetry data available.</p>
          </div>
        )}
      </div>
    </div>
  );
}
