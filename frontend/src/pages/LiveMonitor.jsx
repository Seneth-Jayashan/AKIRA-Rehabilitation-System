import { useState, useEffect } from 'react';
import { Play, Pause, SkipForward, RotateCcw, AlertTriangle, ShieldCheck, Zap, Activity } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, Cell, ReferenceLine } from 'recharts';
import demoData from '../data/demo_data.json';

export default function LiveMonitor() {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [showExplanation, setShowExplanation] = useState(false);
  
  const predictions = demoData.predictions || [];
  const threshold = demoData.threshold || 0.003;
  const currentSample = predictions[currentIndex] || {};
  
  // Animation loop
  useEffect(() => {
    let interval;
    if (isPlaying && currentIndex < predictions.length - 1) {
      interval = setInterval(() => {
        setCurrentIndex(prev => prev + 1);
        setShowExplanation(false); // Hide explanation on new sample
      }, 2000); // 2 second per sample for supervisor explanation
    } else if (currentIndex >= predictions.length - 1) {
      setIsPlaying(false);
    }
    return () => clearInterval(interval);
  }, [isPlaying, currentIndex, predictions.length]);

  const handlePlayPause = () => setIsPlaying(!isPlaying);
  const handleRestart = () => { setCurrentIndex(0); setIsPlaying(false); setShowExplanation(false); };
  const handleNext = () => {
    if (currentIndex < predictions.length - 1) setCurrentIndex(c => c + 1);
    setShowExplanation(false);
  };

  const isAnomalous = currentSample.anomaly_status === 'ANOMALOUS';
  const errorRatio = Math.min(100, (currentSample.reconstruction_error / threshold) * 50); // Scale 50% = threshold

  // Simulated moving waveform window (for visual effect only, representing continuous monitoring)
  const tsData = demoData.timeseries || [];
  const movingWindow = tsData.slice((currentIndex * 2) % (tsData.length - 50), ((currentIndex * 2) % (tsData.length - 50)) + 50);

  return (
    <div className="space-y-6">
      {/* Header and Controls */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 bg-slate-900 p-6 rounded-2xl border border-slate-800 shadow-xl">
        <div>
          <h2 className="text-xl font-bold text-slate-100 flex items-center gap-3">
            <Zap className="text-cyan-400" />
            Explainable AI Live Monitor
          </h2>
          <div className="flex items-center gap-2 mt-2">
            <span className="flex h-2.5 w-2.5 relative">
              <span className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${isPlaying ? 'bg-blue-400' : 'bg-slate-500'}`}></span>
              <span className={`relative inline-flex rounded-full h-2.5 w-2.5 ${isPlaying ? 'bg-blue-500' : 'bg-slate-500'}`}></span>
            </span>
            <span className="text-xs font-semibold text-slate-400 uppercase tracking-widest">
              {isPlaying ? 'Simulation Running' : 'Simulation Paused'}
            </span>
            <span className="ml-4 text-xs bg-amber-950/30 text-amber-500 border border-amber-900/50 px-2 py-0.5 rounded">
              DEMO SIMULATION — NOT REAL SENSOR DATA
            </span>
          </div>
        </div>

        <div className="flex items-center gap-3 bg-slate-950 p-2 rounded-xl border border-slate-800">
          <button onClick={handleRestart} className="p-2 text-slate-400 hover:text-slate-100 hover:bg-slate-800 rounded-lg transition-colors">
            <RotateCcw size={20} />
          </button>
          <button onClick={handlePlayPause} className="px-6 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-lg transition-colors flex items-center gap-2 font-medium">
            {isPlaying ? <Pause size={18} /> : <Play size={18} />}
            {isPlaying ? 'Pause' : 'Play'}
          </button>
          <button onClick={handleNext} disabled={currentIndex >= predictions.length - 1} className="p-2 text-slate-400 hover:text-slate-100 hover:bg-slate-800 rounded-lg transition-colors disabled:opacity-50">
            <SkipForward size={20} />
          </button>
          <div className="pl-4 pr-2 border-l border-slate-800 text-sm font-mono text-slate-400">
            {currentIndex + 1} / {predictions.length}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column: Gauge and Status */}
        <div className="lg:col-span-1 space-y-6">
          <div className="bg-slate-900 p-8 rounded-2xl border border-slate-800 shadow-xl flex flex-col items-center relative overflow-hidden transition-colors duration-500">
            <div className={`absolute inset-0 opacity-10 transition-colors duration-500 ${isAnomalous ? 'bg-red-500' : 'bg-emerald-500'}`}></div>
            
            <h3 className="text-slate-300 font-semibold mb-6 uppercase tracking-wider text-sm relative z-10">
              Reconstruction Error
            </h3>
            
            {/* Circular Gauge Representation */}
            <div className="relative w-48 h-48 flex items-center justify-center mb-6 z-10">
              <svg className="w-full h-full transform -rotate-90" viewBox="0 0 100 100">
                <circle cx="50" cy="50" r="40" stroke="#1e293b" strokeWidth="8" fill="none" />
                <circle 
                  cx="50" cy="50" r="40" 
                  stroke={isAnomalous ? '#ef4444' : '#10b981'} 
                  strokeWidth="8" 
                  fill="none" 
                  strokeDasharray={`${errorRatio * 2.51}, 251.2`} 
                  className="transition-all duration-700 ease-out"
                />
                <circle cx="50" cy="50" r="40" stroke="#f59e0b" strokeWidth="2" fill="none" strokeDasharray="2, 2" strokeDashoffset="-125.6" />
              </svg>
              <div className="absolute flex flex-col items-center">
                <span className="text-3xl font-mono font-bold text-slate-100">
                  {currentSample.reconstruction_error?.toFixed(4)}
                </span>
                <span className="text-xs text-slate-500">MSE</span>
              </div>
            </div>

            <div className="w-full flex justify-between text-xs text-slate-400 mb-6 px-4 z-10 font-mono">
              <span>0.000</span>
              <span className="text-amber-500">Threshold: {threshold.toFixed(4)}</span>
              <span>1.000</span>
            </div>

            <div className={`w-full p-4 rounded-xl border flex items-center gap-3 z-10 transition-colors duration-500 ${
              isAnomalous ? 'bg-red-950/50 border-red-900/50 text-red-400' : 'bg-emerald-950/50 border-emerald-900/50 text-emerald-400'
            }`}>
              {isAnomalous ? <AlertTriangle size={24} className="shrink-0" /> : <ShieldCheck size={24} className="shrink-0" />}
              <div>
                <h4 className="font-bold">{isAnomalous ? 'Anomaly Flagged' : 'Normal Movement'}</h4>
                <p className="text-xs opacity-80 mt-0.5">
                  {isAnomalous 
                    ? 'Error exceeds model threshold.' 
                    : 'Error within acceptable limits.'}
                </p>
              </div>
            </div>
            
            <button 
              onClick={() => setShowExplanation(!showExplanation)}
              className="mt-6 w-full py-2.5 bg-slate-950 hover:bg-slate-800 border border-slate-800 rounded-lg text-sm font-medium text-blue-400 transition-colors z-10"
            >
              {showExplanation ? 'Hide Feature Analysis' : 'Why was this flagged?'}
            </button>
          </div>
        </div>

        {/* Right Column: Telemetry and Explanations */}
        <div className="lg:col-span-2 space-y-6">
          
          {/* Feature Analysis Expander */}
          {showExplanation && (
            <div className="bg-slate-900 p-6 rounded-2xl border border-slate-800 shadow-xl animate-in slide-in-from-top-4 duration-300">
              <h3 className="text-lg font-bold text-slate-100 mb-2 border-b border-slate-800 pb-4">
                Feature-Level Reconstruction Analysis
              </h3>
              <p className="text-sm text-slate-400 mb-6">
                The Autoencoder attempts to compress and reconstruct the input features. Features with the largest squared errors below indicate the model's inability to reconstruct this specific pattern, flagging it as novel or anomalous.
              </p>
              
              <div className="h-[250px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={currentSample.top_feature_errors || []} layout="vertical" margin={{ top: 5, right: 30, left: 80, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} stroke="#1e293b" />
                    <XAxis type="number" stroke="#475569" tick={{fill: '#64748b', fontSize: 12}} />
                    <YAxis type="category" dataKey="feature" stroke="#475569" tick={{fill: '#94a3b8', fontSize: 11}} width={120} />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', color: '#f1f5f9', borderRadius: '0.5rem' }}
                      cursor={{ fill: '#1e293b', opacity: 0.4 }}
                      formatter={(value, name, props) => [
                        `${value.toFixed(5)} (In: ${props.payload.input.toFixed(2)}, Out: ${props.payload.reconstructed.toFixed(2)})`, 
                        'Squared Error'
                      ]}
                    />
                    <Bar dataKey="error" radius={[0, 4, 4, 0]}>
                      {
                        (currentSample.top_feature_errors || []).map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={isAnomalous ? '#ef4444' : '#3b82f6'} />
                        ))
                      }
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* Animated Telemetry */}
          <div className="bg-slate-900 p-6 rounded-2xl border border-slate-800 shadow-xl">
             <div className="flex justify-between items-center mb-6">
              <h3 className="text-lg font-bold text-slate-100 flex items-center gap-3">
                <Activity className="text-slate-400" />
                Raw Waveform Context
              </h3>
              <span className="text-xs text-slate-500 font-mono">Sample Window: 50 ticks</span>
            </div>
            
            <div className="space-y-6">
              <div className="h-[140px] w-full bg-slate-950/50 p-4 rounded-xl border border-slate-800/80">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={movingWindow}>
                    <YAxis domain={[-4, 4]} hide />
                    <Line type="monotone" dataKey="ax" stroke="#ef4444" strokeWidth={2} dot={false} isAnimationActive={false} />
                    <Line type="monotone" dataKey="ay" stroke="#22c55e" strokeWidth={2} dot={false} isAnimationActive={false} />
                    <Line type="monotone" dataKey="az" stroke="#3b82f6" strokeWidth={2} dot={false} isAnimationActive={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              
              <div className="h-[140px] w-full bg-slate-950/50 p-4 rounded-xl border border-slate-800/80">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={movingWindow}>
                    <YAxis domain={[-6, 6]} hide />
                    <Line type="monotone" dataKey="gx" stroke="#ef4444" strokeWidth={2} dot={false} isAnimationActive={false} />
                    <Line type="monotone" dataKey="gy" stroke="#22c55e" strokeWidth={2} dot={false} isAnimationActive={false} />
                    <Line type="monotone" dataKey="gz" stroke="#3b82f6" strokeWidth={2} dot={false} isAnimationActive={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}
