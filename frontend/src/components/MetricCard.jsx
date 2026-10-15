export default function MetricCard({ title, value, subtitle, icon: Icon, colorClass = "blue" }) {
  const colors = {
    blue: "from-blue-500/10 to-transparent text-blue-400 ring-blue-500/30",
    cyan: "from-cyan-500/10 to-transparent text-cyan-400 ring-cyan-500/30",
    emerald: "from-emerald-500/10 to-transparent text-emerald-400 ring-emerald-500/30",
    red: "from-red-500/10 to-transparent text-red-400 ring-red-500/30",
    amber: "from-amber-500/10 to-transparent text-amber-400 ring-amber-500/30",
    slate: "from-slate-500/10 to-transparent text-slate-400 ring-slate-500/30"
  };

  const bgStyle = colors[colorClass] || colors.blue;

  return (
    <div className="bg-slate-900 p-6 rounded-2xl border border-slate-800 shadow-lg relative overflow-hidden group">
      <div className={`absolute top-0 right-0 w-32 h-32 bg-gradient-to-bl ${bgStyle.split(' ')[0]} rounded-bl-full opacity-50 transition-opacity group-hover:opacity-100`}></div>
      <div className="flex justify-between items-start relative z-10">
        <div>
          <p className="text-slate-400 text-sm font-medium uppercase tracking-wider mb-2">{title}</p>
          <h3 className="text-3xl font-bold text-slate-100">{value}</h3>
          {subtitle && <p className="text-slate-500 text-xs mt-2">{subtitle}</p>}
        </div>
        {Icon && (
          <div className={`p-3 rounded-xl bg-slate-950 ring-1 ring-${bgStyle.split('ring-')[1]} text-current`}>
            <Icon size={24} className={bgStyle.split('text-')[1].split(' ')[0] ? `text-${bgStyle.split('text-')[1].split(' ')[0]}` : ''} />
          </div>
        )}
      </div>
    </div>
  );
}
