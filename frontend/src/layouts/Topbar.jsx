import { User } from 'lucide-react';

export default function Topbar({ activeTab }) {
  const formatTitle = (str) => {
    return str.replace(/([A-Z])/g, ' $1').replace(/^./, (s) => s.toUpperCase());
  };

  return (
    <header className="bg-slate-900/40 backdrop-blur-md border-b border-slate-800/50 p-4 px-8 flex justify-between items-center z-20">
      <div className="flex items-center gap-4">
        <h2 className="text-xl font-semibold text-slate-100 tracking-wide">
          {formatTitle(activeTab)}
        </h2>
        <div className="h-4 w-px bg-slate-700"></div>
        <span className="bg-blue-900/30 text-blue-400 border border-blue-800/50 px-2.5 py-1 rounded text-xs font-semibold tracking-wide uppercase">
          Research Prototype
        </span>
      </div>
      
      <div className="flex items-center gap-6">
        <div className="text-right hidden sm:block">
          <p className="text-sm font-medium text-slate-200">Demonstration User</p>
          <p className="text-xs text-slate-500 font-mono">ID: N/A (Not a clinical patient)</p>
        </div>
        <div className="w-10 h-10 rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center text-slate-400 shadow-sm">
          <User size={20} />
        </div>
      </div>
    </header>
  );
}
