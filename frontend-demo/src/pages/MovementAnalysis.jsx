import demoData from '../data/demo_data.json';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { Activity, TriangleAlert } from 'lucide-react';

export default function MovementAnalysis() {
  const features = demoData.features?.slice(0, 30) || []; 

  return (
    <div className="space-y-6">
      <div className="bg-slate-900 p-8 rounded-2xl border border-slate-800 shadow-xl">
        <div className="flex justify-between items-center mb-6 border-b border-slate-800 pb-4">
          <h3 className="text-xl font-bold text-slate-100 flex items-center gap-3">
            <Activity className="text-blue-400" />
            Extracted Movement Features
          </h3>
          <span className="flex items-center gap-2 bg-amber-950/30 text-amber-500 px-3 py-1.5 rounded-lg text-xs font-semibold uppercase tracking-wide border border-amber-900/50">
            <TriangleAlert size={14} />
            ROM: Pending Dual-IMU
          </span>
        </div>
        
        <p className="text-sm text-slate-400 mb-8 leading-relaxed max-w-4xl bg-slate-950 p-4 rounded-xl border border-slate-800 shadow-inner">
          Displays extracted time-domain kinematics per movement trial. 
          <strong className="text-slate-200"> Note:</strong> True spatial wrist orientation (Absolute Range of Motion, Flexion/Extension angles) 
          cannot be accurately calculated without the synchronized Forearm + Hand hardware prototype.
        </p>

        {features.length > 0 ? (
          <div className="h-[450px] w-full bg-slate-950/50 p-6 rounded-2xl border border-slate-800/80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={features} margin={{ top: 20, right: 30, left: -20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#1e293b" />
                <XAxis dataKey="trial_id" hide />
                <YAxis yAxisId="left" orientation="left" stroke="#475569" tick={{fill: '#64748b', fontSize: 12}} />
                <YAxis yAxisId="right" orientation="right" stroke="#475569" tick={{fill: '#64748b', fontSize: 12}} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', color: '#f1f5f9', borderRadius: '0.5rem' }}
                  itemStyle={{ fontSize: 13 }}
                  cursor={{ fill: '#1e293b', opacity: 0.4 }}
                />
                <Legend wrapperStyle={{ paddingTop: '20px' }} />
                <Bar yAxisId="left" dataKey="acc_mag_mean" fill="#3b82f6" name="Mean Accel Mag (g)" radius={[4, 4, 0, 0]} />
                <Bar yAxisId="right" dataKey="gyro_mag_max" fill="#8b5cf6" name="Max Gyro Mag (rad/s)" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <div className="h-[450px] flex items-center justify-center text-slate-500 border border-dashed border-slate-800 rounded-xl bg-slate-950/50">
            Feature data unavailable.
          </div>
        )}
      </div>
    </div>
  );
}
