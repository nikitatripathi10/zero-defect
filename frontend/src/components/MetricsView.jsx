import React from "react";
import { CheckCircle2, BarChart2, Award, TrendingUp, Zap } from "lucide-react";
import { BorderBeam } from "./ui/BorderBeam";
import { ScrollReveal } from "./ui/ScrollReveal";
import { TiltCard } from "./ui/TiltCard";

const MetricBar = ({ value, max = 100, color }) => {
  const pct = typeof value === "number" ? (value / max) * 100 : parseFloat(value) * 100;
  return (
    <div className="w-full h-1.5 rounded-full bg-white/[0.06] overflow-hidden mt-3">
      <div
        className="h-full rounded-full transition-all duration-1000"
        style={{
          width: `${Math.min(pct, 100)}%`,
          background: color,
          boxShadow: `0 0 8px ${color}60`,
        }}
      />
    </div>
  );
};

export const MetricsView = () => {
  const metrics = [
    {
      label: "Precision",
      value: "94.8%",
      raw: 94.8,
      max: 100,
      desc: "Ultra-low false-positive alarm rate",
      color: "#3b82f6",
      gradient: "from-blue-400 to-cyan-400",
    },
    {
      label: "Recall",
      value: "92.4%",
      raw: 92.4,
      max: 100,
      desc: "Comprehensive defect capture coverage",
      color: "#7c3aed",
      gradient: "from-violet-400 to-purple-400",
    },
    {
      label: "F1 Score",
      value: "0.936",
      raw: 0.936,
      max: 1,
      desc: "Harmonic precision-recall equilibrium",
      color: "#10b981",
      gradient: "from-emerald-400 to-teal-400",
    },
    {
      label: "IoU Overlap",
      value: "0.880",
      raw: 0.880,
      max: 1,
      desc: "Sub-pixel overlap with ground-truth masks",
      color: "#f59e0b",
      gradient: "from-amber-400 to-orange-400",
    },
  ];

  const confusionMatrix = [
    {
      label: "True Positives",
      count: "148,290 px",
      pct: "92.4%",
      note: "Correctly localized defect pixels",
      color: "emerald",
    },
    {
      label: "False Positives",
      count: "8,140 px",
      pct: "5.2%",
      note: "Surface texture false alarms",
      color: "amber",
    },
    {
      label: "False Negatives",
      count: "12,180 px",
      pct: "7.6%",
      note: "Faint hairline fractures missed",
      color: "red",
    },
    {
      label: "True Negatives",
      count: "2,453,030 px",
      pct: "99.7%",
      note: "Correctly classified clean steel",
      color: "blue",
    },
  ];

  const colorMap = {
    emerald: "text-emerald-400 bg-emerald-500/10 border-emerald-500/20",
    amber: "text-amber-400 bg-amber-500/10 border-amber-500/20",
    red: "text-red-400 bg-red-500/10 border-red-500/20",
    blue: "text-blue-400 bg-blue-500/10 border-blue-500/20",
  };

  const compliance = [
    {
      text: "Sub-Millimeter Edge Accuracy",
      detail: "Calibrated with auto-adaptive Canny hysteresis thresholding per surface type.",
    },
    {
      text: "Multi-Angle GLCM Matrices",
      detail: "0°, 45°, 90°, and 135° directional variance modeling across all steel grades.",
    },
    {
      text: "Automated Anomaly Triage",
      detail: "Real-time bounding box extraction with severity classification for line rejects.",
    },
    {
      text: "Morphological Noise Suppression",
      detail: "Erosion and dilation kernels tuned to defect class geometry and surface texture.",
    },
  ];

  return (
    <div className="py-8 px-4 sm:px-8 max-w-7xl mx-auto">
      {/* Header */}
      <ScrollReveal>
        <div className="mb-10">
          <div className="flex flex-wrap items-center gap-3 mb-3">
            <h2 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
              Quantitative Quality Verification
            </h2>
            <span className="text-[10px] px-2.5 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/25 text-emerald-400 font-mono tracking-widest">
              CALIBRATED BENCHMARKS
            </span>
          </div>
          <p className="text-sm text-slate-400 max-w-2xl">
            Pixel-level verification calibrated against ASTM E45 and ISO 9001 manufacturing tolerance criteria.
          </p>
        </div>
      </ScrollReveal>

      {/* Metric Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-10">
        {metrics.map((m, idx) => (
          <ScrollReveal key={idx} direction="up" delay={idx * 0.08}>
            <TiltCard
              className="relative overflow-hidden rounded-2xl p-6 bg-slate-900/55 border border-white/[0.07] backdrop-blur-xl shadow-xl card-glow-hover"
              glowColor={`${m.color}30`}
            >
              <BorderBeam size={160} duration={10 + idx * 2} delay={idx * 2} colorFrom={m.color} colorTo="#8b5cf6" />
              <div className="text-[10px] font-bold uppercase tracking-widest text-slate-500 mb-2">
                {m.label}
              </div>
              <div className={`text-3xl font-extrabold bg-clip-text text-transparent bg-gradient-to-r ${m.gradient}`}>
                {m.value}
              </div>
              <div className="mt-2 text-[11px] text-slate-500 leading-relaxed">{m.desc}</div>
              <MetricBar value={m.raw} max={m.max} color={m.color} />
            </TiltCard>
          </ScrollReveal>
        ))}
      </div>

      {/* Bottom section */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Confusion Matrix */}
        <ScrollReveal className="lg:col-span-7" direction="left">
          <div className="rounded-2xl p-6 bg-slate-900/55 border border-white/[0.07] backdrop-blur-xl h-full">
            <h3 className="text-xs font-bold uppercase tracking-widest text-slate-300 mb-5 flex items-center gap-2">
              <BarChart2 className="w-4 h-4 text-blue-400" />
              <span>Pixel-Level Confusion Matrix</span>
              <span className="ml-auto text-[10px] font-mono text-slate-500">FACTORY RUN #492</span>
            </h3>
            <div className="grid grid-cols-2 gap-3">
              {confusionMatrix.map((item, idx) => (
                <div
                  key={idx}
                  className={`p-4 rounded-xl border ${colorMap[item.color]} bg-opacity-10 transition-all duration-200 hover:scale-[1.02]`}
                >
                  <div className={`text-[10px] font-mono mb-1 ${colorMap[item.color].split(" ")[0]}`}>
                    {item.label}
                  </div>
                  <div className="text-lg font-extrabold text-white font-mono">{item.count}</div>
                  <div className={`text-xs font-bold mt-0.5 ${colorMap[item.color].split(" ")[0]}`}>
                    {item.pct}
                  </div>
                  <div className="text-[11px] text-slate-500 mt-1">{item.note}</div>
                </div>
              ))}
            </div>
          </div>
        </ScrollReveal>

        {/* Compliance */}
        <ScrollReveal className="lg:col-span-5" direction="right">
          <div className="rounded-2xl p-6 bg-slate-900/55 border border-white/[0.07] backdrop-blur-xl h-full flex flex-col">
            <h3 className="text-xs font-bold uppercase tracking-widest text-slate-300 mb-5 flex items-center gap-2">
              <Award className="w-4 h-4 text-cyan-400" />
              <span>Detection Capabilities</span>
            </h3>

            <div className="space-y-4 flex-1">
              {compliance.map((item, idx) => (
                <div key={idx} className="flex items-start gap-3">
                  <div className="mt-0.5 w-5 h-5 rounded-full bg-emerald-500/15 border border-emerald-500/25 flex items-center justify-center shrink-0">
                    <CheckCircle2 className="w-3 h-3 text-emerald-400" />
                  </div>
                  <div>
                    <div className="text-xs font-semibold text-slate-200">{item.text}</div>
                    <div className="text-[11px] text-slate-500 mt-0.5 leading-relaxed">{item.detail}</div>
                  </div>
                </div>
              ))}
            </div>

            <div className="mt-6 p-4 rounded-xl bg-gradient-to-r from-blue-500/8 to-violet-500/8 border border-blue-500/15 flex items-center justify-between">
              <div>
                <div className="text-[10px] text-slate-500 uppercase tracking-widest">Production Line Status</div>
                <div className="text-sm font-bold text-emerald-400 font-mono mt-0.5 flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse inline-block" />
                  FULLY OPERATIONAL
                </div>
              </div>
              <Zap className="w-5 h-5 text-blue-400 opacity-60" />
            </div>
          </div>
        </ScrollReveal>
      </div>
    </div>
  );
};
