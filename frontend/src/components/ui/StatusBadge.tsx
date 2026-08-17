"use client";

import React from "react";
import {
  CheckCircle2,
  Clock,
  AlertCircle,
  Pause,
  Play,
  ShieldCheck,
  XCircle,
  Activity,
} from "lucide-react";

export type StatusType =
  | "QUEUED"
  | "RUNNING"
  | "COMPLETED"
  | "FAILED"
  | "PAUSED"
  | "WAITING_FOR_REVIEW"
  | "CANCELLED"
  | "DRAFT"
  | "READY"
  | string;

interface StatusBadgeProps {
  status: StatusType;
  className?: string;
  size?: "sm" | "md";
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({
  status,
  className = "",
  size = "md",
}) => {
  const norm = (status || "").toUpperCase().replace(/\s+/g, "_");

  let colorClasses = "bg-neutral-800 text-neutral-300 border-neutral-700";
  let Icon = Clock;
  let label = norm.replace(/_/g, " ");

  if (norm === "COMPLETED" || norm === "HEALTHY" || norm === "READY") {
    colorClasses = "bg-emerald-950/50 text-emerald-300 border-emerald-800/40";
    Icon = CheckCircle2;
  } else if (norm === "RUNNING") {
    colorClasses = "bg-indigo-950/50 text-indigo-300 border-indigo-800/40";
    Icon = Play;
  } else if (norm === "WAITING_FOR_REVIEW" || norm === "REVIEW") {
    colorClasses = "bg-amber-950/50 text-amber-300 border-amber-800/40";
    Icon = ShieldCheck;
  } else if (norm === "PAUSED") {
    colorClasses = "bg-neutral-800 text-neutral-400 border-neutral-700";
    Icon = Pause;
  } else if (norm === "FAILED" || norm === "ERROR") {
    colorClasses = "bg-rose-950/50 text-rose-300 border-rose-800/40";
    Icon = AlertCircle;
  } else if (norm === "CANCELLED") {
    colorClasses = "bg-neutral-900 text-neutral-500 border-neutral-800";
    Icon = XCircle;
  } else if (norm === "QUEUED") {
    colorClasses = "bg-cyan-950/50 text-cyan-300 border-cyan-800/40";
    Icon = Clock;
  }

  const sizeClasses =
    size === "sm"
      ? "text-[10px] px-1.5 py-0.5 gap-1"
      : "text-xs px-2.5 py-1 gap-1.5";

  return (
    <span
      className={`inline-flex items-center rounded-full font-mono font-medium border ${colorClasses} ${sizeClasses} ${className}`}
    >
      <Icon className={size === "sm" ? "w-2.5 h-2.5 shrink-0" : "w-3 h-3 shrink-0"} />
      <span>{label}</span>
    </span>
  );
};
