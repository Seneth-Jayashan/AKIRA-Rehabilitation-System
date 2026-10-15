import { LayoutDashboard, LineChart, Activity, ShieldAlert, History, Info, MonitorPlay, Cpu, BarChart2 } from 'lucide-react';
import axios from 'axios';
import { useState, useEffect } from 'react';

const NAV_ITEMS = [
  { id: 'dashboard', icon: LayoutDashboard, label: '1. Overview' },
  { id: 'anomalies', icon: ShieldAlert, label: '2. Autoencoder Anomaly' },
  { id: 'analysis', icon: Activity, label: '3. Movement Classifiers' },
  { id: 'compare', icon: Cpu, label: '4. Compare Classifiers' },
  { id: 'explorer', icon: LineChart, label: '5. Sensor Data Explorer' },
  { id: 'sessions', icon: History, label: '6. Prediction History' },
  { id: 'status', icon: BarChart2, label: '7. Model Performance' },
  { id: 'about', icon: Info, label: '8. Reports & About' },
];

export default function Sidebar({ activeTab, setActiveTab }) {
  const [sysStatus, setSysStatus] = useState(null);
  
  useEffect(() => {
    axios.get('http://localhost:8000/api/health')
      .then(res => setSysStatus(res.data))
      .catch(() => setSysStatus({status: 'offline'}));
  }, []);

  return (
    <aside className="w-72 bg-slate-900 border-r border-slate-800 flex flex-col shadow-2xl z-20">
      <div className="p-6">
        <h1 className="text-3xl font-extrabold tracking-wider text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-cyan-400 flex items-center gap-3">
          <Activity className="text-blue-500" size={32} />
          AKIRA
        </h1>
        <p className="text-slate-400 text-xs mt-2 uppercase tracking-widest font-semibold ml-11">
          Unified Multi-Model
        </p>
      </div>
      
      <nav className="flex-1 px-4 py-4 space-y-2 overflow-y-auto">
        {NAV_ITEMS.map((item) => {
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              className={`w-full flex items-center gap-3 px-4 py-3.5 rounded-xl transition-all duration-200 group ${
                isActive 
                  ? 'bg-slate-800 text-cyan-400 border border-slate-700/50 shadow-inner' 
                  : 'text-slate-400 hover:bg-slate-800/40 hover:text-slate-200 border border-transparent'
              }`}
            >
              <item.icon size={20} className={isActive ? 'text-cyan-400' : 'text-slate-500 group-hover:text-slate-400'} />
              <span className="font-medium text-sm">{item.label}</span>
            </button>
          );
        })}
      </nav>
      
      <div className="p-6 border-t border-slate-800/80 bg-slate-900/50">
        <div className="bg-slate-950 border border-slate-800/80 p-4 rounded-xl text-xs space-y-3 shadow-inner">
          <div className="flex justify-between items-center text-slate-400">
            <span className="font-medium">API Gateway</span>
            <span className={sysStatus?.status === 'ok' ? "text-emerald-500" : "text-red-500"}>{sysStatus?.status === 'ok' ? 'ONLINE' : 'OFFLINE'}</span>
          </div>
          <div className="flex justify-between items-center text-slate-400">
            <span className="font-medium">RF / CNN / XGB</span>
            <span className={sysStatus?.rf_loaded ? "text-blue-400" : "text-red-500"}>{sysStatus?.rf_loaded ? 'READY' : 'MISSING'}</span>
          </div>
          <div className="flex justify-between items-center text-slate-400">
            <span className="font-medium">Autoencoder</span>
            <span className={sysStatus?.ae_loaded ? "text-amber-400" : "text-red-500"}>{sysStatus?.ae_loaded ? 'READY' : 'MISSING'}</span>
          </div>
        </div>
      </div>
    </aside>
  );
}
