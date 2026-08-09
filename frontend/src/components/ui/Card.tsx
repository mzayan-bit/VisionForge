import React from "react";

export interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: "default" | "glass" | "bordered";
  hoverable?: boolean;
}

export const Card: React.FC<CardProps> = ({
  children,
  variant = "default",
  hoverable = false,
  className = "",
  ...props
}) => {
  const variantStyles = {
    default: "bg-[#111111] border border-white/10 shadow-sm",
    glass: "glass-panel shadow-md",
    bordered: "bg-transparent border border-white/15",
  };

  const hoverStyles = hoverable
    ? "transition-all duration-200 hover:border-white/25 hover:bg-[#161616] hover:shadow-lg hover:shadow-black/40"
    : "";

  return (
    <div
      className={`rounded-lg p-5 ${variantStyles[variant]} ${hoverStyles} ${className}`}
      {...props}
    >
      {children}
    </div>
  );
};

export const CardHeader: React.FC<React.HTMLAttributes<HTMLDivElement>> = ({
  children,
  className = "",
  ...props
}) => (
  <div className={`mb-4 pb-3 border-b border-white/10 ${className}`} {...props}>
    {children}
  </div>
);

export const CardTitle: React.FC<React.HTMLAttributes<HTMLHeadingElement>> = ({
  children,
  className = "",
  ...props
}) => (
  <h3 className={`text-lg font-semibold text-white ${className}`} {...props}>
    {children}
  </h3>
);

export const CardBody: React.FC<React.HTMLAttributes<HTMLDivElement>> = ({
  children,
  className = "",
  ...props
}) => (
  <div className={className} {...props}>
    {children}
  </div>
);

export const CardContent = CardBody;

export const CardFooter: React.FC<React.HTMLAttributes<HTMLDivElement>> = ({
  children,
  className = "",
  ...props
}) => (
  <div className={`mt-4 pt-3 border-t border-white/10 ${className}`} {...props}>
    {children}
  </div>
);
