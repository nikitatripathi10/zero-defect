import React, { useEffect, useRef, useState } from "react";

/**
 * NumberCounter — animates a numeric value up from 0 on mount
 * Inspired by Magic UI's number-ticker pattern
 */
export const NumberCounter = ({
  value,
  duration = 2000,
  className = "",
  suffix = "",
  prefix = "",
  decimals = 0,
}) => {
  const [display, setDisplay] = useState("0");
  const startTime = useRef(null);
  const rafRef = useRef(null);

  const parseTarget = (val) => {
    const str = String(val).replace(/[^0-9.]/g, "");
    return parseFloat(str) || 0;
  };

  useEffect(() => {
    const target = parseTarget(value);
    startTime.current = null;

    const step = (timestamp) => {
      if (!startTime.current) startTime.current = timestamp;
      const progress = Math.min((timestamp - startTime.current) / duration, 1);
      // Ease-out cubic
      const eased = 1 - Math.pow(1 - progress, 3);
      const current = eased * target;
      setDisplay(
        decimals > 0
          ? current.toFixed(decimals)
          : Math.floor(current).toLocaleString()
      );
      if (progress < 1) {
        rafRef.current = requestAnimationFrame(step);
      } else {
        // Final value from original string
        setDisplay(
          decimals > 0
            ? target.toFixed(decimals)
            : String(value).replace(/[^0-9.%+]/g, "").replace(/^\d+/, (m) =>
                parseInt(m, 10).toLocaleString()
              )
        );
      }
    };

    rafRef.current = requestAnimationFrame(step);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [value, duration, decimals]);

  return (
    <span className={className}>
      {prefix}{display}{suffix}
    </span>
  );
};

