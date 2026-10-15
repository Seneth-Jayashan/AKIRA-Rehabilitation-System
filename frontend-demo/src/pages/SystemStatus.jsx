import { Server, Database, Brain, Activity, Code } from 'lucide-react';

export default function SystemStatus() {
  const Card = ({ icon: Icon, title, status, desc, color }) => {
    const colors = {
      emerald: "bg-emerald-950/30 border-emerald-900/50 text-emerald-400 icon-emerald-400",
      blue: "bg-blue-950/30 border-blue-900/50 text-blue-400 icon-blue-400",
      amber: "bg-amber-950/30 border-amber-900/50 text-amber-400 icon-amber-400",
      slate: "bg-slate-900/50 border-slate-800 text-slate-400 icon-slate-500"
    };
    
    const style = colors[color];
    
    return (
      <div className={`p-6 border rounded-xl flex gap-5 ${style.split(' text-')[0]}`}>
        <Icon className={`${style.split(' icon-')[1]} mt-1 shrink-0`} size={28} />
        <div>
          <div className="flex items-center gap-3 mb-2">
            <h4 className={`font-bold text-lg ${style.split(' border-')[0].replace('bg-', 'text-').replace('/30', '')}`}>{title}</h4>
            <span className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider border ${style.replace('bg-', 'border-').split(' ')[1]}`}>
              {status}
            </span>
          </div>
          <p className="text-sm opacity-80 leading-relaxed text-slate-300">
            {desc}
          </p>
        </div>
      </div>
    );
  };

  return (
    <div className="max-w-4xl">
      <div className="bg-slate-900 p-8 rounded-2xl shadow-xl border border-slate-800 relative overflow-hidden">
        <h2 className="text-2xl font-bold text-slate-100 mb-8">AKIRA System Architecture Status</h2>
        
        <div className="space-y-4 relative z-10">
          <Card 
            icon={Database}
            title="Dataset Standardization"
            status="Complete"
            desc="Data pipeline successfully built. Preprocessing, filtering, and segmentation fully implemented on public WatchHand single-IMU data."
            color="emerald"
          />
          <Card 
            icon={Brain}
            title="Anomaly Detection Prototype"
            status="Complete"
            desc="Autoencoder model successfully trained offline to detect irregular movements based on single-IMU kinematics. Thresholds calculated automatically."
            color="emerald"
          />
          <Card 
            icon={Code}
            title="Supervisor Demo Frontend"
            status="Complete"
            desc="React/Vite dashboard securely connected to local offline CSV demonstration payloads to protect source environment integrity."
            color="blue"
          />
          <Card 
            icon={Activity}
            title="Dual-IMU ROM Tracking"
            status="Pending Hardware"
            desc="True relative wrist orientation (flexion/extension/deviation) requires synchronized Forearm + Hand IMUs. Pending physical deployment."
            color="amber"
          />
          <Card 
            icon={Server}
            title="Clinical Diagnostic Engine"
            status="Pending Data"
            desc="Longitudinal patient recovery scoring cannot be claimed until the model is trained on confirmed post-ORIF patient telemetry."
            color="slate"
          />
        </div>
      </div>
    </div>
  );
}
