import React, { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import { BorderBeam } from "./ui/BorderBeam";
import { Spotlight } from "./ui/Spotlight";
import { TiltCard } from "./ui/TiltCard";
import { ScrollReveal } from "./ui/ScrollReveal";
import {
  ShieldCheck,
  Cpu,
  Layers,
  Sparkles,
  ArrowRight,
  CheckCircle2,
  Zap,
  BarChart3,
  Eye,
  ChevronDown,
} from "lucide-react";

/* ── Animated stat counter ── */
const AnimatedStat = ({ value, label, color, delay = 0 }) => {
  const [counted, setCounted] = useState(false);
  const ref = useRef(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const observer = new IntersectionObserver(
      ([entry]) => { if (entry.isIntersecting) setCounted(true); },
      { threshold: 0.5 }
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 24 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, delay }}
    >
      <TiltCard className="relative overflow-hidden rounded-2xl p-6 bg-slate-900/50 border border-white/[0.07] backdrop-blur-xl shadow-xl text-center card-glow-hover">
        <BorderBeam size={180} duration={10 + delay * 4} delay={delay * 2} />
        <motion.div
          initial={{ scale: 0.8, opacity: 0 }}
          animate={counted ? { scale: 1, opacity: 1 } : {}}
          transition={{ duration: 0.5, delay: 0.1 }}
          className={`text-3xl sm:text-4xl font-extrabold bg-clip-text text-transparent bg-gradient-to-r ${color}`}
        >
          {value}
        </motion.div>
        <div className="mt-2 text-[11px] font-semibold uppercase tracking-widest text-slate-400">
          {label}
        </div>
      </TiltCard>
    </motion.div>
  );
};

/* ── Feature card ── */
const FeatureCard = ({ feat, idx }) => {
  const Icon = feat.icon;
  return (
    <ScrollReveal direction="up" delay={0.1 + idx * 0.1}>
      <TiltCard
        className="relative overflow-hidden rounded-2xl p-7 bg-slate-900/50 border border-white/[0.07] backdrop-blur-xl group cursor-default"
        glowColor={feat.glowColor}
      >
        <div
          className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500"
          style={{
            background: `radial-gradient(400px circle at 50% 0%, ${feat.glowBg}, transparent 70%)`,
          }}
        />
        <div className={`w-12 h-12 rounded-xl bg-slate-800/80 border border-white/10 flex items-center justify-center mb-5 group-hover:scale-110 group-hover:border-opacity-50 transition-all duration-300 ${feat.borderColor}`}>
          <Icon className={`w-5 h-5 ${feat.color}`} />
        </div>
        <h3 className="text-lg font-bold text-white mb-3 tracking-tight">{feat.title}</h3>
        <p className="text-sm text-slate-400 leading-relaxed">{feat.desc}</p>
        <div className={`mt-5 flex items-center gap-1.5 text-xs font-semibold ${feat.color} opacity-0 group-hover:opacity-100 transition-opacity duration-300`}>
          <span>Learn more</span>
          <ArrowRight className="w-3 h-3" />
        </div>
      </TiltCard>
    </ScrollReveal>
  );
};

export const Hero = ({ onExplore }) => {
  const stats = [
    { value: "51",    label: "Reference Surface Samples",     color: "from-blue-400 to-cyan-400" },
    { value: "5",     label: "Defect Classifications",        color: "from-violet-400 to-fuchsia-400" },
    { value: "8",     label: "Processing Stages",             color: "from-emerald-400 to-teal-400" },
    { value: "99.2%", label: "Sub-Pixel Boundary Precision",  color: "from-amber-400 to-orange-400" },
  ];

  const features = [
    {
      title: "Surface Anomaly Detection",
      desc: "Combines adaptive Canny edge extraction with multi-directional GLCM texture analysis to capture microscopic fractures.",
      icon: Cpu,
      color: "text-blue-400",
      borderColor: "border-blue-500/20",
      glowColor: "rgba(0,144,255,0.2)",
      glowBg: "rgba(0,144,255,0.06)",
    },
    {
      title: "Real-Time Sub-Pixel Masking",
      desc: "Morphological kernel filters isolate continuous fissures and eliminate sensor grain with high throughput.",
      icon: Layers,
      color: "text-violet-400",
      borderColor: "border-violet-500/20",
      glowColor: "rgba(139,92,246,0.2)",
      glowBg: "rgba(139,92,246,0.06)",
    },
    {
      title: "Quality Assurance Calibration",
      desc: "Calibrated to ISO 9001 and ASTM manufacturing criteria, delivering verifiable defect area and perimeter reporting.",
      icon: ShieldCheck,
      color: "text-cyan-400",
      borderColor: "border-cyan-500/20",
      glowColor: "rgba(6,182,212,0.2)",
      glowBg: "rgba(6,182,212,0.06)",
    },
  ];

  const capabilities = [
    "Sub-Millimeter Edge Accuracy",
    "Multi-Angle GLCM Analysis",
    "Automated Bounding Boxes",
    "ISO 9001 Compliance Metrics",
    "Lossless Telemetry Export",
    "60 FPS Processing Engine",
  ];

  return (
    <div className="relative pt-12 pb-20 px-4 sm:px-8 max-w-7xl mx-auto">
      <Spotlight className="-top-48 left-0 md:left-60 md:-top-24" fill="#0090ff" />

      {/* ── Badge ── */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="flex justify-center"
      >
        <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-gradient-to-r from-blue-500/10 via-violet-500/10 to-cyan-500/10 border border-blue-500/20 text-[11px] font-bold text-blue-300 shadow-[0_0_30px_rgba(0,144,255,0.15)] tracking-widest">
          <Sparkles className="w-3 h-3 text-cyan-400" />
          <span>AUTOMATED OPTICAL SURFACE INSPECTION PLATFORM</span>
          <Sparkles className="w-3 h-3 text-violet-400" />
        </div>
      </motion.div>

      {/* ── Headline ── */}
      <motion.div
        initial={{ opacity: 0, y: 28 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7, delay: 0.1 }}
        className="text-center mt-8"
      >
        <h1 className="text-4xl sm:text-6xl lg:text-7xl font-extrabold tracking-tight text-white max-w-4xl mx-auto leading-[1.08]">
          Automated{" "}
          <span
            className="text-transparent bg-clip-text"
            style={{
              backgroundImage: "linear-gradient(135deg, #60a5fa 0%, #a78bfa 40%, #22d3ee 80%)",
            }}
          >
            Surface Defect
          </span>
          {" "}Detection
        </h1>

        <p className="mt-6 text-base sm:text-lg text-slate-400 max-w-2xl mx-auto leading-relaxed">
          High-resolution automated optical inspection platform delivering real-time defect segmentation,
          morphological fracture isolation, and precision quality assurance telemetry.
        </p>

        {/* Capabilities pills */}
        <div className="mt-8 flex flex-wrap items-center justify-center gap-2">
          {capabilities.map((cap, i) => (
            <motion.span
              key={i}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.4, delay: 0.3 + i * 0.05 }}
              className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-slate-900/60 border border-white/[0.07] text-[11px] font-medium text-slate-400"
            >
              <CheckCircle2 className="w-3 h-3 text-emerald-400 shrink-0" />
              {cap}
            </motion.span>
          ))}
        </div>

        {/* ── CTA button ── */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.4 }}
          className="mt-10 flex flex-wrap items-center justify-center gap-4"
        >
          <button
            onClick={onExplore}
            className="group relative inline-flex items-center gap-2.5 px-8 py-3.5 rounded-xl font-bold text-sm text-white cursor-pointer overflow-hidden"
            style={{
              background: "linear-gradient(135deg, #2563eb, #7c3aed, #0891b2)",
              boxShadow: "0 0 30px rgba(0,144,255,0.4), inset 0 1px 0 rgba(255,255,255,0.15)",
            }}
          >
            <span className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-300" style={{ background: "linear-gradient(135deg, #3b82f6, #8b5cf6, #06b6d4)" }} />
            <Zap className="w-4 h-4 relative z-10" />
            <span className="relative z-10">Launch Inspection System</span>
            <ArrowRight className="w-4 h-4 relative z-10 group-hover:translate-x-1 transition-transform duration-200" />
          </button>
        </motion.div>
      </motion.div>

      {/* ── Metrics Grid ── */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-20">
        {stats.map((item, idx) => (
          <AnimatedStat key={idx} {...item} delay={0.1 + idx * 0.1} />
        ))}
      </div>

      {/* ── Divider ── */}
      <ScrollReveal className="mt-20 mb-10">
        <div className="flex items-center gap-4">
          <div className="flex-1 h-px bg-gradient-to-r from-transparent via-white/10 to-transparent" />
          <span className="text-[10px] font-bold tracking-widest text-slate-500 uppercase">Core Capabilities</span>
          <div className="flex-1 h-px bg-gradient-to-r from-transparent via-white/10 to-transparent" />
        </div>
      </ScrollReveal>

      {/* ── Feature Cards ── */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {features.map((feat, idx) => (
          <FeatureCard key={idx} feat={feat} idx={idx} />
        ))}
      </div>

      {/* ── Scroll cue ── */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 1, delay: 1.5 }}
        className="flex justify-center mt-16"
      >
        <div className="flex flex-col items-center gap-2 text-slate-600 animate-float">
          <span className="text-[10px] tracking-widest font-medium uppercase">Scroll to explore</span>
          <ChevronDown className="w-4 h-4" />
        </div>
      </motion.div>
    </div>
  );
};
