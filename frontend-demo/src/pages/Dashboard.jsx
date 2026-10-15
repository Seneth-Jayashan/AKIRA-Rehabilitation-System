import demoData from '../data/demo_data.json';
import MetricCard from '../components/MetricCard';
import { Activity, AlertTriangle, Database, Cpu, Info } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

export default function Dashboard() {
  const anomaliesCount = demoData.predictions?.filter(p => p.anomaly_status === 'ANOMALOUS').length || 0;
  const analyzedSamples = demoData.predictions?.length || 0;
  
  // Aggregate some metrics for the chart
  const summaryData = demoData.predictions?.slice(0, 50).reduce((acc, curr) => {
    const ex = curr.exercise_original || 'Unknown';
    const found = acc.find(a => a.name === ex);
    if (found) {
      found.total += 1;
      if (curr.anomaly_status === 'ANOMALOUS') found.anomalies += 1;
    } else {
      acc.push({ name: ex, total: 1, anomalies: curr.anomaly_status === 'ANOMALOUS' ? 1 : 0 });
    }
    return acc;
  }, []) || [];

  return (
    <div className="space-y-8">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard 
          title="Analyzed Samples" 
          value={analyzedSamples} 
          subtitle="From WatchHand Dataset"
          icon={Activity} 
          colorClass="blue" 
        />
        <MetricCard 
          title="Anomalies Detected" 
          value={anomaliesCount} 
          subtitle="Algorithmically flagged"
          icon={AlertTriangle} 
          colorClass="red" 
        />
        <MetricCard 
          title="Data Source" 
          value="Public Set" 
          subtitle="02_STANDARDIZED_DATA"
          icon={Database} 
          colorClass="emerald" 
        />
        <MetricCard 
          title="AI Engine" 
          value="Online" 
          subtitle="autoencoder.keras"
          icon={Cpu} 
          colorClass="cyan" 
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 bg-slate-900 p-6 rounded-2xl border border-slate-800 shadow-lg">
          <h3 className="text-lg font-bold text-slate-100 mb-6 border-b border-slate-800 pb-4">
            Recent Analysis Distribution (Subset)
          </h3>
          <div className="h-[300px]">
            {summaryData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={summaryData} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#1e293b" />
                  <XAxis dataKey="name" stroke="#64748b" tick={{fill: '#94a3b8', fontSize: 12}} angle={-15} textAnchor="end" />
                  <YAxis stroke="#64748b" tick={{fill: '#94a3b8', fontSize: 12}} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', color: '#f1f5f9', borderRadius: '0.5rem' }}
                    cursor={{ fill: '#1e293b', opacity: 0.4 }}
                  />
                  <Bar dataKey="total" name="Total Samples" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="anomalies" name="Flagged Anomalies" fill="#ef4444" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-full flex items-center justify-center text-slate-500 border border-dashed border-slate-800 rounded-xl bg-slate-950/50">
                Chart Data Unavailable
              </div>
            )}
          </div>
        </div>

        <div className="bg-slate-900 p-6 rounded-2xl border border-slate-800 shadow-lg relative overflow-hidden">
          <div className="absolute top-0 right-0 p-16 bg-blue-500/5 rounded-bl-full pointer-events-none"></div>
          <h3 className="text-lg font-bold text-slate-100 mb-4 flex items-center gap-2">
            <Info className="text-blue-400" size={20} /> Important Notice
          </h3>
          <div className="space-y-4 text-sm text-slate-400 leading-relaxed">
            <p>
              This dashboard is rendering local demonstration data exported directly from the AKIRA pipeline outputs.
            </p>
            <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 shadow-inner">
              <p className="font-semibold text-amber-400 mb-1">Single-IMU Prototype</p>
              <p>The current system utilizes single-IMU smartwatch data. Dual-IMU relative ROM analysis is disabled.</p>
            </div>
            <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 shadow-inner">
              <p className="font-semibold text-cyan-400 mb-1">Non-Clinical Data</p>
              <p>Displayed anomalies are algorithmic reconstruction errors, not medical pathology. Actual post-ORIF patient data is required for clinical validation.</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
