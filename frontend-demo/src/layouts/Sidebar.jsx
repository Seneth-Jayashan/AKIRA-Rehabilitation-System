import { LayoutDashboard, LineChart, Activity, ShieldAlert, History, Info, MonitorPlay } from 'lucide-react';

const NAV_ITEMS = [
  { id: 'dashboard', icon: LayoutDashboard, label: 'Overview' },
  { id: 'monitor', icon: MonitorPlay, label: 'Live Monitor' },
  { id: 'explorer', icon: LineChart, label: 'IMU Explorer' },
  { id: 'analysis', icon: Activity, label: 'Feature Explanation' },
  { id: 'anomalies', icon: ShieldAlert, label: 'Prediction History' },
  { id: 'sessions', icon: History, label: 'Session History' },
  { id: 'status', icon: Info, label: 'Model Information' },
];

export default function Sidebar({ activeTab, setActiveTab }) {
  return (
    <aside className="w-72 bg-slate-900 border-r border-slate-800 flex flex-col shadow-2xl z-20">
      <div className="p-6">
        <h1 className="text-3xl font-extrabold tracking-wider text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-cyan-400 flex items-center gap-3">
          <Activity className="text-blue-500" size={32} />
          AKIRA
        </h1>
        <p className="text-slate-400 text-xs mt-2 uppercase tracking-widest font-semibold ml-11">
          Rehabilitation
        </p>
      </div>
      
      <nav className="flex-1 px-4 py-6 space-y-2 overflow-y-auto">
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
          <div className="flex justify-between items-center">
            <span className="text-slate-500 font-medium uppercase tracking-wider">Mode</span>
            <span className="text-blue-400 font-mono font-medium">Single-IMU Demo</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-slate-500 font-medium uppercase tracking-wider">Hardware</span>
            <span className="text-red-400 font-mono font-medium flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-red-500"></span> Disconnected
            </span>
          </div>
        </div>
      </div>
    </aside>
  );
}
