import React, { useEffect, useRef, useState } from "react";

/**
 * ScrollReveal — reveals children with animation when scrolled into view
 * Inspired by Motion (Framer Motion) scroll animations
 */
export const ScrollReveal = ({
  children,
  className = "",
  direction = "up", // "up" | "down" | "left" | "right" | "fade"
  delay = 0,
  duration = 0.6,
  threshold = 0.15,
  once = true,
}) => {
  const ref = useRef(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setVisible(true);
          if (once) observer.unobserve(el);
        } else if (!once) {
          setVisible(false);
        }
      },
      { threshold }
    );

    observer.observe(el);
    return () => observer.disconnect();
  }, [threshold, once]);

  const getInitial = () => {
    switch (direction) {
      case "up":    return "translateY(32px)";
      case "down":  return "translateY(-32px)";
      case "left":  return "translateX(32px)";
      case "right": return "translateX(-32px)";
      default:      return "translateY(0)";
    }
  };

  return (
    <div
      ref={ref}
      className={className}
      style={{
        opacity: visible ? 1 : 0,
        transform: visible ? "translate(0, 0)" : getInitial(),
        transition: `opacity ${duration}s cubic-bezier(0.16,1,0.3,1) ${delay}s, transform ${duration}s cubic-bezier(0.16,1,0.3,1) ${delay}s`,
        willChange: "opacity, transform",
      }}
    >
      {children}
    </div>
  );
};

