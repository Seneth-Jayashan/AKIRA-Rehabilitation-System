import { useState, useEffect } from 'react';
import axios from 'axios';
import MetricCard from '../components/MetricCard';
import { Activity, ShieldAlert, Cpu, BarChart2 } from 'lucide-react';

export default function Dashboard() {
  const [modelInfo, setModelInfo] = useState(null);
  
  useEffect(() => {
    axios.get('http://localhost:8000/api/model-info').then(res => setModelInfo(res.data)).catch(console.error);
  }, []);

  return (
    <div className="space-y-8">
      <div className="flex items-center gap-3 border-b border-slate-800 pb-4 mb-8">
        <Activity className="text-blue-400" size={28} />
        <h2 className="text-3xl font-black text-white tracking-tight">System Overview</h2>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard 
          title="Autoencoder Threshold" 
          value={modelInfo?.ae?.threshold ? modelInfo.ae.threshold.toFixed(5) : 'N/A'} 
          subtitle="Model 1 (95th%)"
          icon={ShieldAlert} 
          trend="+0.0%"
          trendLabel="baseline"
        />
        <MetricCard 
          title="Random Forest Acc." 
          value={modelInfo?.rf?.metrics ? (modelInfo.rf.metrics.accuracy * 100).toFixed(1) + '%' : 'N/A'} 
          subtitle="Model 2"
          icon={Cpu} 
        />
        <MetricCard 
          title="1D CNN Acc." 
          value={modelInfo?.cnn?.metrics ? (modelInfo.cnn.metrics.accuracy * 100).toFixed(1) + '%' : 'N/A'} 
          subtitle="Model 3"
          icon={BarChart2} 
        />
        <MetricCard 
          title="XGBoost Acc." 
          value={modelInfo?.xgb?.metrics ? (modelInfo.xgb.metrics.accuracy * 100).toFixed(1) + '%' : 'N/A'} 
          subtitle="Model 4"
          icon={Activity} 
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-8">
        <div className="bg-slate-900/80 backdrop-blur p-8 rounded-3xl border border-slate-800 shadow-xl">
          <h3 className="text-lg font-bold text-slate-100 flex items-center gap-3 mb-6">
            <ShieldAlert className="text-amber-400" /> Model 1: Autoencoder
          </h3>
          <p className="text-slate-400 text-sm leading-relaxed mb-6">
            Processes raw accelerometer and gyroscope generic IMU data. Flags anomalous execution via reconstruction Mean Squared Error (MSE).
          </p>
          <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 flex justify-between items-center">
            <span className="text-slate-400 text-sm">Verified Threshold (95th%)</span>
            <span className="text-amber-400 font-mono font-bold text-lg">{modelInfo?.ae?.threshold ? modelInfo.ae.threshold.toFixed(6) : 'N/A'}</span>
          </div>
        </div>

        <div className="bg-slate-900/80 backdrop-blur p-8 rounded-3xl border border-slate-800 shadow-xl">
          <h3 className="text-lg font-bold text-slate-100 flex items-center gap-3 mb-6">
            <Cpu className="text-cyan-400" /> Quaternion Classifiers
          </h3>
          <p className="text-slate-400 text-sm leading-relaxed mb-6">
            Models 2, 3, and 4 predict discrete wrist movements from 3DCDataset absolute spatial quaternions using distinct machine-learning architectures.
          </p>
          <div className="grid grid-cols-2 gap-3 text-sm">
             {['NEUTRAL', 'RADIAL_DEVIATION', 'WRIST_FLEXION', 'ULNAR_DEVIATION'].map(c => (
               <div key={c} className="bg-slate-950 p-3 rounded-lg border border-slate-800 flex items-center justify-center text-center text-xs font-medium text-slate-300">
                 {c.replace('_', ' ')}
               </div>
             ))}
          </div>
        </div>
      </div>
    </div>
  );
}
