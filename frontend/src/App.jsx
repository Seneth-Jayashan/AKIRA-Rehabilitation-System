import { useState, useEffect } from 'react';
import Sidebar from './layouts/Sidebar';
import Topbar from './layouts/Topbar';
import Dashboard from './pages/Dashboard';
import DataExplorer from './pages/DataExplorer';
import MovementAnalysis from './pages/MovementAnalysis';
import AnomalyDetection from './pages/AnomalyDetection';
import SessionHistory from './pages/SessionHistory';
import SystemStatus from './pages/SystemStatus';

export default function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [predictionHistory, setPredictionHistory] = useState([]);

  const addHistory = (entry) => {
    setPredictionHistory(prev => [{ id: Date.now(), timestamp: new Date().toLocaleTimeString(), ...entry }, ...prev]);
  };

  const renderContent = () => {
    switch (activeTab) {
      case 'dashboard': return <Dashboard />;
      case 'anomalies': return <AnomalyDetection addHistory={addHistory} />;
      case 'analysis': return <MovementAnalysis key="analysis" addHistory={addHistory} mode="single" />;
      case 'compare': return <MovementAnalysis key="compare" addHistory={addHistory} mode="compare" />;
      case 'explorer': return <DataExplorer />;
      case 'sessions': return <SessionHistory history={predictionHistory} />;
      case 'status': return <SystemStatus key="status" mode="performance" />;
      case 'about': return <SystemStatus key="about" mode="about" />;
      default: return <Dashboard />;
    }
  };

  return (
    <div className="flex h-screen bg-slate-950 text-slate-200 overflow-hidden selection:bg-cyan-900 selection:text-cyan-50">
      <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} />
      
      <div className="flex-1 flex flex-col relative z-0">
        <div className="absolute top-[-20%] left-[-10%] w-[50%] h-[50%] bg-blue-900/10 rounded-full blur-[120px] pointer-events-none"></div>
        <div className="absolute bottom-[-20%] right-[-10%] w-[40%] h-[40%] bg-cyan-900/10 rounded-full blur-[100px] pointer-events-none"></div>
        
        <Topbar activeTab={activeTab} />
        
        <main className="flex-1 overflow-y-auto p-6 lg:p-10 z-10">
          <div className="max-w-7xl mx-auto animate-in fade-in duration-500">
            {renderContent()}
          </div>
        </main>
      </div>
    </div>
  );
}
