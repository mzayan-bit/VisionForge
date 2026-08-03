import React from "react";

export interface SectionHeaderProps {
  title: string;
  badge?: string;
  action?: React.ReactNode;
  className?: string;
}

export const SectionHeader: React.FC<SectionHeaderProps> = ({
  title,
  badge,
  action,
  className = "",
}) => {
  return (
    <div className={`flex items-center justify-between mb-3 ${className}`}>
      <div className="flex items-center gap-2.5">
        <h2 className="text-xs font-semibold text-neutral-400 uppercase tracking-widest font-geist">
          {title}
        </h2>
        {badge && (
          <span className="text-[10px] px-1.5 py-0.5 rounded bg-neutral-800 text-neutral-400 border border-white/10 font-mono">
            {badge}
          </span>
        )}
      </div>
      {action && <div className="text-xs text-neutral-400">{action}</div>}
    </div>
  );
};
