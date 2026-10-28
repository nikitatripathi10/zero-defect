import React from "react";
import { cn } from "../../lib/utils";

/**
 * TextShimmer — animated shimmer gradient text
 * Inspired by Magic UI's text-shimmer pattern
 */
export const TextShimmer = ({
  children,
  className = "",
  duration = 3,
  spread = 2,
  as: Tag = "span",
}) => {
  return (
    <Tag
      className={cn(
        "relative inline-block bg-clip-text text-transparent",
        "bg-gradient-to-r from-white/70 via-white via-blue-400 via-violet-400 to-white/70",
        "bg-[length:200%_auto] animate-shimmer",
        className
      )}
      style={{
        backgroundImage: `linear-gradient(
          90deg,
          rgba(255,255,255,0.6) 0%,
          rgba(255,255,255,0.9) 25%,
          rgba(0,144,255,1) 38%,
          rgba(139,92,246,1) 50%,
          rgba(6,182,212,1) 62%,
          rgba(255,255,255,0.9) 75%,
          rgba(255,255,255,0.6) 100%
        )`,
        backgroundSize: "200% auto",
        animation: `shimmer ${duration}s linear infinite`,
      }}
    >
      {children}
    </Tag>
  );
};

