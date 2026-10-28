import React from "react";

export const BorderBeam = ({
  className = "",
  size = 300,
  duration = 10,
  borderWidth = 1.5,
  colorFrom = "#0090ff",
  colorTo = "#8b5cf6",
  delay = 0,
}) => {
  return (
    <div
      style={{
        "--size": `${size}px`,
        "--duration": `${duration}s`,
        "--color-from": colorFrom,
        "--color-to": colorTo,
        "--delay": `-${delay}s`,
      }}
      className={`pointer-events-none absolute inset-0 rounded-[inherit] [border:calc(var(--border-width)*1px)_solid_transparent] 
      ![mask-clip:padding-box,border-box] ![mask-composite:intersect] 
      [mask:linear-gradient(transparent,transparent),linear-gradient(white,white)]
      after:absolute after:aspect-square after:w-[calc(var(--size)*1px)] 
      after:animate-[border-beam_var(--duration)_infinite_linear] 
      after:[animation-delay:var(--delay)] 
      after:[background:linear-gradient(to_left,var(--color-from),var(--color-to),transparent)] 
      after:[offset-anchor:100%_50%] after:[offset-path:rect(0_auto_auto_0_round_calc(var(--size)*1px))] ${className}`}
    />
  );
};
