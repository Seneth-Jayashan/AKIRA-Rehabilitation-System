import { History, CalendarX } from 'lucide-react';

export default function SessionHistory() {
  return (
    <div className="space-y-6">
      <div className="bg-slate-900 p-8 rounded-2xl border border-slate-800 shadow-xl min-h-[500px] flex flex-col relative overflow-hidden">
        <div className="absolute top-0 right-0 p-32 bg-slate-800/20 rounded-full blur-3xl pointer-events-none"></div>
        
        <h3 className="text-xl font-bold text-slate-100 flex items-center gap-3 mb-2 relative z-10">
          <History className="text-slate-400" />
          Longitudinal Session History
        </h3>
        
        <div className="flex-1 flex flex-col items-center justify-center relative z-10 text-center max-w-md mx-auto">
          <div className="w-24 h-24 bg-slate-950 rounded-full flex items-center justify-center border border-slate-800 mb-6 shadow-inner">
            <CalendarX size={40} className="text-slate-600" />
          </div>
          <h4 className="text-lg font-semibold text-slate-300 mb-2">Patient Data Unavailable</h4>
          <p className="text-slate-500 text-sm leading-relaxed">
            Longitudinal session tracking requires repeated session associations. The current public dataset (WatchHand) does not contain multi-session patient recovery timelines.
          </p>
          <div className="mt-8 px-4 py-2 bg-slate-950 border border-slate-800 rounded-lg text-xs font-semibold uppercase tracking-wider text-slate-600">
            Feature Planned for Clinical Phase
          </div>
        </div>
      </div>
    </div>
  );
}
