import React from "react";

export const AuroraBackground = ({
  className = "",
  children,
  showRadialGradient = true,
  ...props
}) => {
  return (
    <div
      className={`relative flex flex-col min-h-screen bg-[#030712] text-slate-100 transition-bg ${className}`}
      {...props}
    >
      {/* Animated grid overlay */}
      <div
        className="pointer-events-none fixed inset-0 z-0 animate-grid-fade"
        style={{
          backgroundImage: `
            linear-gradient(rgba(0,144,255,0.04) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0,144,255,0.04) 1px, transparent 1px)
          `,
          backgroundSize: "48px 48px",
        }}
      />

      {/* Aurora gradient layer */}
      <div className="pointer-events-none fixed inset-0 z-0 overflow-hidden">
        <div
          className={`
            [--dark-gradient:repeating-linear-gradient(100deg,rgba(2,6,23,0)_0%,rgba(0,144,255,0.10)_15%,rgba(139,92,246,0.14)_25%,rgba(6,182,212,0.12)_35%,rgba(2,6,23,0)_50%)]
            [--aurora:repeating-linear-gradient(100deg,#0090ff_10%,#8b5cf6_18%,#06b6d4_26%,#a855f7_34%,#0090ff_42%)]
            [background-image:var(--dark-gradient),var(--aurora)]
            [background-size:300%,_200%]
            [background-position:50%_50%,50%_50%]
            filter blur-[80px]
            after:content-[""] after:absolute after:inset-0
            after:[background-image:var(--dark-gradient),var(--aurora)]
            after:[background-size:200%,_100%]
            after:animate-aurora after:[background-attachment:fixed]
            after:mix-blend-difference
            pointer-events-none
            absolute -inset-[10px] opacity-50 will-change-transform
          `}
        />
        {/* Radial vignette */}
        {showRadialGradient && (
          <div className="absolute inset-0 [background-image:radial-gradient(ellipse_80%_60%_at_50%_0%,transparent_20%,#030712_85%)]" />
        )}
        {/* Ambient orbs */}
        <div className="absolute top-[-20%] left-[-10%] w-[600px] h-[600px] rounded-full bg-blue-600/8 blur-[120px] animate-pulse-slow" />
        <div className="absolute top-[10%] right-[-15%] w-[500px] h-[500px] rounded-full bg-violet-600/8 blur-[120px] animate-pulse-slow" style={{ animationDelay: "-4s" }} />
        <div className="absolute bottom-[0%] left-[30%] w-[400px] h-[400px] rounded-full bg-cyan-600/6 blur-[100px] animate-pulse-slow" style={{ animationDelay: "-8s" }} />
      </div>

      <div className="relative z-10 w-full">{children}</div>
    </div>
  );
};
