"use client";

import React from "react";
import { AlertCircle, RefreshCw, Info } from "lucide-react";
import { Button } from "./Button";

interface ErrorStateProps {
  title?: string;
  message?: string;
  requestId?: string;
  onRetry?: () => void;
  className?: string;
}

export const ErrorState: React.FC<ErrorStateProps> = ({
  title = "Failed to load resource",
  message = "An unexpected error occurred while communicating with the VisionForge backend API.",
  requestId,
  onRetry,
  className = "",
}) => {
  return (
    <div
      className={`p-8 rounded-2xl bg-neutral-900/60 border border-rose-500/20 text-center space-y-3 max-w-lg mx-auto ${className}`}
    >
      <div className="w-12 h-12 rounded-xl bg-rose-500/10 border border-rose-500/30 flex items-center justify-center text-rose-400 mx-auto">
        <AlertCircle className="w-6 h-6" />
      </div>

      <div className="space-y-1">
        <h4 className="text-sm font-semibold text-white">{title}</h4>
        <p className="text-xs text-neutral-400 leading-relaxed font-sans">{message}</p>
      </div>

      {requestId && (
        <div className="inline-flex items-center gap-1 text-[11px] font-mono text-neutral-500 bg-neutral-950 px-2 py-1 rounded border border-white/5">
          <Info className="w-3 h-3" />
          <span>Request ID: {requestId}</span>
        </div>
      )}

      {onRetry && (
        <div className="pt-2">
          <Button
            variant="secondary"
            size="sm"
            onClick={onRetry}
            className="flex items-center gap-1.5 mx-auto text-xs text-neutral-200"
          >
            <RefreshCw className="w-3.5 h-3.5" /> Retry Request
          </Button>
        </div>
      )}
    </div>
  );
};
