import React from "react";

interface StatusCardProps {
  id?: string;
  title: string;
  subtitle: string;
  status: string;
  statusType?: "success" | "warning" | "info";
  iconText: string;
}

export function StatusCard({
  id,
  title,
  subtitle,
  status,
  statusType = "success",
  iconText,
}: StatusCardProps) {
  const statusColors = {
    success: "bg-emerald-500/10 text-emerald-400 border-emerald-500/20",
    warning: "bg-amber-500/10 text-amber-400 border-amber-500/20",
    info: "bg-cyan-500/10 text-cyan-400 border-cyan-500/20",
  };

  return (
    <div
      id={id}
      className="p-6 rounded-xl bg-slate-900/60 border border-slate-800 backdrop-blur-sm hover:border-slate-700 transition-all duration-200"
    >
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-3">
          <div className="h-10 w-10 rounded-lg bg-slate-800 border border-slate-700/50 flex items-center justify-center font-mono text-cyan-400 text-sm font-semibold">
            {iconText}
          </div>
          <div>
            <h3 className="font-semibold text-slate-100">{title}</h3>
            <p className="text-xs text-slate-400 mt-0.5">{subtitle}</p>
          </div>
        </div>
        <span
          className={`text-xs font-mono px-2.5 py-1 rounded-full border ${statusColors[statusType]}`}
        >
          {status}
        </span>
      </div>
    </div>
  );
}
