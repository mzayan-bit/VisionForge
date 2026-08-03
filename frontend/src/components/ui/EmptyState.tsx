import React from "react";

export interface EmptyStateProps {
  icon?: React.ReactNode;
  title: string;
  description: string;
  action?: React.ReactNode;
  className?: string;
}

export const EmptyState: React.FC<EmptyStateProps> = ({
  icon,
  title,
  description,
  action,
  className = "",
}) => {
  return (
    <div
      className={`flex flex-col items-center justify-center p-8 md:p-12 text-center rounded-xl border border-dashed border-white/15 bg-neutral-900/40 ${className}`}
    >
      {icon && (
        <div className="w-12 h-12 rounded-full bg-neutral-800/80 border border-white/10 flex items-center justify-center text-blue-400 mb-4 shadow-sm">
          {icon}
        </div>
      )}
      <h3 className="text-lg font-medium text-white mb-1 font-geist">{title}</h3>
      <p className="text-sm text-neutral-400 max-w-md mb-6 leading-relaxed">{description}</p>
      {action && <div>{action}</div>}
    </div>
  );
};
