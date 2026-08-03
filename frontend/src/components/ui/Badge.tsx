import React from "react";

export interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: "success" | "warning" | "error" | "info" | "neutral";
  size?: "sm" | "md";
  dot?: boolean;
}

export const Badge: React.FC<BadgeProps> = ({
  children,
  variant = "neutral",
  size = "md",
  dot = false,
  className = "",
  ...props
}) => {
  const variantStyles = {
    success: "bg-emerald-500/10 text-emerald-400 border-emerald-500/25",
    warning: "bg-amber-500/10 text-amber-400 border-amber-500/25",
    error: "bg-rose-500/10 text-rose-400 border-rose-500/25",
    info: "bg-blue-500/10 text-blue-400 border-blue-500/25",
    neutral: "bg-neutral-800/80 text-neutral-300 border-white/10",
  };

  const dotColors = {
    success: "bg-emerald-400",
    warning: "bg-amber-400",
    error: "bg-rose-400",
    info: "bg-blue-400",
    neutral: "bg-neutral-400",
  };

  const sizeStyles = {
    sm: "text-[10px] px-1.5 py-0.5 gap-1 font-mono uppercase tracking-wider",
    md: "text-xs px-2.5 py-1 gap-1.5 font-medium",
  };

  return (
    <span
      className={`inline-flex items-center rounded-md border ${variantStyles[variant]} ${sizeStyles[size]} ${className}`}
      {...props}
    >
      {dot && <span className={`w-1.5 h-1.5 rounded-full ${dotColors[variant]}`} />}
      {children}
    </span>
  );
};
