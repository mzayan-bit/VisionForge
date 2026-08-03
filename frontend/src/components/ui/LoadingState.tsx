import React from "react";

export interface LoadingStateProps {
  label?: string;
  className?: string;
}

export const LoadingState: React.FC<LoadingStateProps> = ({
  label = "Loading data...",
  className = "",
}) => {
  return (
    <div className={`flex flex-col items-center justify-center p-8 text-center ${className}`}>
      <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin mb-3" />
      <span className="text-xs font-mono text-neutral-400 uppercase tracking-widest">{label}</span>
    </div>
  );
};

export const Skeleton: React.FC<{ className?: string }> = ({ className = "h-4 w-full" }) => {
  return <div className={`animate-pulse bg-neutral-800/60 rounded ${className}`} />;
};
