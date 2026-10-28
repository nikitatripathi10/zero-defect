import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Eye, Layers, Sparkles, Sliders, ChevronRight } from "lucide-react";
import { ScrollReveal } from "./ui/ScrollReveal";
import { TiltCard } from "./ui/TiltCard";

const stages = [
  {
    step: 1,
    title: "Grayscale Conversion & Gaussian Filtration",
    code: "cv2.GaussianBlur(gray, (5, 5), sigma=1.2)",
    desc: "Removes chrominance interference while smoothing high-frequency sensor noise without blurring defect boundaries. The kernel size is adaptively tuned to the surface material's grain profile.",
    formula: "G(x, y) = 1/(2πσ²) · e^(-(x²+y²)/(2σ²))",
    gradient: "from-blue-500/15 to-cyan-500/5",
    accent: "#3b82f6",
    accentLight: "rgba(59,130,246,0.1)",
    border: "border-blue-500/25",
    tag: "STAGE 01",
  },
  {
    step: 2,
    title: "Contrast Limited Adaptive Histogram (CLAHE)",
    code: "cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))",
    desc: "Enhances localized pixel contrast across steel grain variations, preventing over-amplification in uniform regions. Tile-based processing preserves fine structural detail.",
    formula: "P_CLAHE(i) = clip_limit(h(i)) / ∑ h(k)",
    gradient: "from-violet-500/15 to-purple-500/5",
    accent: "#7c3aed",
    accentLight: "rgba(124,58,237,0.1)",
    border: "border-violet-500/25",
    tag: "STAGE 02",
  },
  {
    step: 3,
    title: "Sub-Pixel Gradient & Auto-Canny Edges",
    code: "canny_edges(img, sigma=0.33, aperture=3)",
    desc: "Computes Sobel gradient magnitude followed by non-maximum suppression and hysteresis thresholding. The σ-adaptive threshold eliminates manual calibration per surface type.",
    formula: "G_mag = |Gx| + |Gy|  ·  θ = arctan(Gy / Gx)",
    gradient: "from-emerald-500/15 to-teal-500/5",
    accent: "#10b981",
    accentLight: "rgba(16,185,129,0.1)",
    border: "border-emerald-500/25",
    tag: "STAGE 03",
  },
  {
    step: 4,
    title: "Morphological Component Cleaning",
    code: "cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)",
    desc: "Performs mathematical morphology: erosion removes single-pixel salt noise; dilation bridges fragmented scratch paths. Structuring element shape is optimized per defect class.",
    formula: "A • B = (A ⊕ B) ⊖ B",
    gradient: "from-amber-500/15 to-orange-500/5",
    accent: "#f59e0b",
    accentLight: "rgba(245,158,11,0.1)",
    border: "border-amber-500/25",
    tag: "STAGE 04",
  },
  {
    step: 5,
    title: "GLCM Texture Anomaly Heatmap",
    code: "graycomatrix(levels=256, distances=[1,3], angles=[0,45,90])",
    desc: "Calculates Gray-Level Co-occurrence Matrix properties across four angular directions. Contrast, Dissimilarity, Homogeneity, and Local Entropy reveal invisible sub-surface stress.",
    formula: "Contrast = ∑ |i − j|² · P(i, j)",
    gradient: "from-fuchsia-500/15 to-pink-500/5",
    accent: "#d946ef",
    accentLight: "rgba(217,70,239,0.1)",
    border: "border-fuchsia-500/25",
    tag: "STAGE 05",
  },
];

export const PipelineView = () => {
  const [activeStage, setActiveStage] = useState(0);
  const stage = stages[activeStage];

  return (
    <div className="py-8 px-4 sm:px-8 max-w-7xl mx-auto">
      {/* Header */}
      <ScrollReveal>
        <div className="mb-10">
          <div className="flex flex-wrap items-center gap-3 mb-3">
            <h2 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
              Optical Inspection Pipeline
            </h2>
            <span className="text-[10px] px-2.5 py-1 rounded-full bg-blue-500/10 border border-blue-500/25 text-blue-400 font-mono tracking-widest">
              5-STAGE PIPELINE
            </span>
          </div>
          <p className="text-sm text-slate-400 max-w-2xl">
            Multi-stage surface inspection pipeline from raw sensor capture to calibrated defect masks.
          </p>
        </div>
      </ScrollReveal>

      {/* Stage selector */}
      <ScrollReveal delay={0.1}>
        <div className="flex flex-wrap gap-2 mb-8">
          {stages.map((stg, idx) => (
            <button
              key={idx}
              onClick={() => setActiveStage(idx)}
              className={`group relative px-4 py-2.5 rounded-xl text-xs font-semibold tracking-wide transition-all duration-200 cursor-pointer flex items-center gap-2 ${
                activeStage === idx
                  ? "text-white border"
                  : "bg-slate-900/50 border border-white/[0.06] text-slate-400 hover:text-white hover:bg-slate-800/60"
              }`}
              style={
                activeStage === idx
                  ? {
                      background: stg.accentLight,
                      borderColor: `${stg.accent}50`,
                      boxShadow: `0 0 20px ${stg.accent}25`,
                    }
                  : {}
              }
            >
              <span
                className="w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold transition-colors"
                style={
                  activeStage === idx
                    ? { background: stg.accent, color: "white" }
                    : { background: "rgba(255,255,255,0.08)", color: "#94a3b8" }
                }
              >
                {stg.step}
              </span>
              <span className="hidden sm:inline">{stg.title.split(" ")[0]} Stage</span>
              <span className="sm:hidden">{stg.step}</span>
            </button>
          ))}
        </div>
      </ScrollReveal>

      {/* Active stage detail */}
      <AnimatePresence mode="wait">
        <motion.div
          key={activeStage}
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -16 }}
          transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
          className={`rounded-2xl p-6 sm:p-8 bg-gradient-to-br ${stage.gradient} bg-slate-900/70 border ${stage.border} backdrop-blur-2xl shadow-2xl`}
        >
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
            {/* Left: Details */}
            <div className="lg:col-span-7">
              <div className="inline-flex items-center gap-2 text-[10px] font-mono mb-3 tracking-widest" style={{ color: stage.accent }}>
                <Sparkles className="w-3 h-3" />
                <span>{stage.tag} · MATHEMATICAL TRANSFORM</span>
              </div>
              <h3 className="text-xl sm:text-2xl font-bold text-white mb-3 tracking-tight">
                {stage.title}
              </h3>
              <p className="text-sm text-slate-300 leading-relaxed mb-6">
                {stage.desc}
              </p>

              {/* Code block */}
              <div className="rounded-xl bg-slate-950/80 border border-white/[0.07] overflow-hidden mb-4">
                <div className="px-4 py-2 border-b border-white/[0.06] flex items-center gap-2">
                  <div className="flex gap-1.5">
                    <div className="w-2.5 h-2.5 rounded-full bg-red-500/60" />
                    <div className="w-2.5 h-2.5 rounded-full bg-yellow-500/60" />
                    <div className="w-2.5 h-2.5 rounded-full bg-green-500/60" />
                  </div>
                  <span className="text-[10px] text-slate-500 font-mono uppercase tracking-widest ml-1">Pipeline Operation</span>
                </div>
                <div className="p-4 font-mono text-xs text-blue-300">
                  <code>{stage.code}</code>
                </div>
              </div>

              {/* Formula block */}
              <div className="p-4 rounded-xl bg-slate-950/40 border border-white/[0.05] font-mono text-xs">
                <div className="text-[10px] text-slate-500 uppercase font-sans font-bold mb-2 tracking-widest">Mathematical Function</div>
                <div className="font-semibold" style={{ color: stage.accent }}>{stage.formula}</div>
              </div>
            </div>

            {/* Right: Visualizer */}
            <div className="lg:col-span-5">
              <div className="text-[10px] font-mono text-slate-500 mb-2 uppercase tracking-widest">Output Buffer Preview</div>
              <div className="relative aspect-video rounded-xl overflow-hidden bg-slate-950 border border-white/[0.08] shadow-xl">
                {/* Procedural viz */}
                <div className="w-full h-full flex items-center justify-center relative overflow-hidden">
                  {/* Animated grid */}
                  <div
                    className="absolute inset-0 opacity-30"
                    style={{
                      backgroundImage: `radial-gradient(${stage.accent}55 1px, transparent 1px)`,
                      backgroundSize: "14px 14px",
                    }}
                  />
                  {/* Scan animation */}
                  <motion.div
                    key={activeStage}
                    initial={{ top: "0%" }}
                    animate={{ top: "105%" }}
                    transition={{ duration: 2.5, repeat: Infinity, ease: "linear" }}
                    className="absolute left-0 right-0 h-px opacity-80"
                    style={{
                      background: `linear-gradient(90deg, transparent, ${stage.accent}, transparent)`,
                      boxShadow: `0 0 8px ${stage.accent}`,
                    }}
                  />
                  {/* Center info */}
                  <div className="relative z-10 text-center p-4">
                    <div
                      className="text-xs font-mono font-bold mb-1"
                      style={{ color: stage.accent }}
                    >
                      TRANSFORM #{stage.tag.split(" ")[1]} RESOLVED
                    </div>
                    <div className="text-[11px] text-slate-500">
                      Latency: 1.84 ms · 60 FPS
                    </div>
                  </div>
                </div>
              </div>

              {/* Stage navigation */}
              <div className="flex items-center justify-between mt-4">
                <button
                  onClick={() => setActiveStage(Math.max(0, activeStage - 1))}
                  disabled={activeStage === 0}
                  className="px-3 py-1.5 rounded-lg bg-slate-900/60 border border-white/[0.06] text-xs text-slate-400 hover:text-white disabled:opacity-30 transition-all cursor-pointer disabled:cursor-not-allowed"
                >
                  ← Previous
                </button>
                <div className="flex gap-1">
                  {stages.map((_, i) => (
                    <button
                      key={i}
                      onClick={() => setActiveStage(i)}
                      className="w-2 h-2 rounded-full transition-all cursor-pointer"
                      style={
                        i === activeStage
                          ? { background: stage.accent, transform: "scale(1.4)" }
                          : { background: "rgba(255,255,255,0.15)" }
                      }
                    />
                  ))}
                </div>
                <button
                  onClick={() => setActiveStage(Math.min(stages.length - 1, activeStage + 1))}
                  disabled={activeStage === stages.length - 1}
                  className="px-3 py-1.5 rounded-lg bg-slate-900/60 border border-white/[0.06] text-xs text-slate-400 hover:text-white disabled:opacity-30 transition-all cursor-pointer disabled:cursor-not-allowed"
                >
                  Next →
                </button>
              </div>
            </div>
          </div>
        </motion.div>
      </AnimatePresence>
    </div>
  );
};
