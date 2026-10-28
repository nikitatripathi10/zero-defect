import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { BorderBeam } from "./ui/BorderBeam";
import { ScrollReveal } from "./ui/ScrollReveal";
import { TiltCard } from "./ui/TiltCard";
import {
  Play, Sliders, AlertTriangle, RefreshCw,
  FileText, Target, ChevronDown, Zap,
} from "lucide-react";

const categories = [
  {
    id: "scratch",
    name: "Scratch Defects",
    desc: "Longitudinal surface scarring on oiled hot-rolled steel coils.",
    icon: "⚡",
  },
  {
    id: "small_defect",
    name: "Micro Cavities",
    desc: "Sub-millimeter pitting and localized surface indentations.",
    icon: "🔵",
  },
  {
    id: "wipe_crack",
    name: "Wipe Cracks",
    desc: "Transverse thermal stress fractures from high-speed rolling.",
    icon: "〰️",
  },
  {
    id: "sdi",
    name: "SDI Steel Surface",
    desc: "Complex industrial surface anomalies and oxidation blemishes.",
    icon: "🔶",
  },
  {
    id: "spdi",
    name: "SPDI Pipe Telemetry",
    desc: "Curved cylindrical pipe seam fissures and slag inclusions.",
    icon: "🔴",
  },
];

const sampleRegions = [
  { id: 1, type: "Linear Fracture", area: "842 px", conf: "98.4%", bbox: "[X:114, Y:82, W:240, H:42]", severity: "Critical" },
  { id: 2, type: "Micro Scratch",   area: "318 px", conf: "94.1%", bbox: "[X:290, Y:210, W:110, H:24]", severity: "Major" },
  { id: 3, type: "Surface Pit",     area: "164 px", conf: "89.7%", bbox: "[X:412, Y:145, W:45, H:38]",  severity: "Minor" },
];

const severityStyle = {
  Critical: "bg-red-500/15 text-red-400 border border-red-500/25",
  Major:    "bg-amber-500/15 text-amber-400 border border-amber-500/25",
  Minor:    "bg-blue-500/15 text-blue-400 border border-blue-500/25",
};

export const InspectorView = () => {
  const [selectedCategory, setSelectedCategory] = useState("scratch");
  const [blurKernel, setBlurKernel] = useState(5);
  const [useClahe, setUseClahe] = useState(true);
  const [sensitivity, setSensitivity] = useState(65);
  const [minArea, setMinArea] = useState(120);
  const [isScanning, setIsScanning] = useState(false);
  const [scanComplete, setScanComplete] = useState(true);
  const [scanCount, setScanCount] = useState(0);

  const handleRunInspection = () => {
    setIsScanning(true);
    setScanComplete(false);
    setTimeout(() => {
      setIsScanning(false);
      setScanComplete(true);
      setScanCount((c) => c + 1);
    }, 1400);
  };

  return (
    <div className="py-8 px-4 sm:px-8 max-w-7xl mx-auto">
      {/* Header */}
      <ScrollReveal>
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-10">
          <div>
            <div className="flex flex-wrap items-center gap-3 mb-2">
              <h2 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
                Surface Inspection Console
              </h2>
              <span className="text-[10px] px-2.5 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/25 text-emerald-400 font-mono tracking-widest flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                ACTIVE SENSOR
              </span>
            </div>
            <p className="text-sm text-slate-400">
              Optical surface evaluation with real-time defect segmentation and anomaly scoring.
            </p>
          </div>

          <button
            onClick={handleRunInspection}
            disabled={isScanning}
            className="inline-flex items-center gap-2.5 px-6 py-3 rounded-xl text-white font-bold text-sm cursor-pointer disabled:opacity-60 disabled:cursor-not-allowed transition-all duration-200 self-start md:self-auto shrink-0"
            style={{
              background: isScanning
                ? "rgba(59,130,246,0.2)"
                : "linear-gradient(135deg, #2563eb, #7c3aed)",
              boxShadow: isScanning ? "none" : "0 0 25px rgba(0,144,255,0.35)",
            }}
          >
            {isScanning ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin" />
                <span>Analyzing Surface Pixels…</span>
              </>
            ) : (
              <>
                <Zap className="w-4 h-4" />
                <span>Run Inspection</span>
              </>
            )}
          </button>
        </div>
      </ScrollReveal>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Controls */}
        <div className="lg:col-span-4 flex flex-col gap-5">
          {/* Category selector */}
          <ScrollReveal direction="left">
            <div className="rounded-2xl p-5 bg-slate-900/55 border border-white/[0.07] backdrop-blur-xl">
              <h3 className="text-[10px] font-bold uppercase tracking-widest text-slate-400 mb-4 flex items-center gap-2">
                <Sliders className="w-3.5 h-3.5 text-blue-400" />
                Surface Defect Class
              </h3>
              <div className="flex flex-col gap-2">
                {categories.map((cat) => (
                  <button
                    key={cat.id}
                    onClick={() => setSelectedCategory(cat.id)}
                    className={`text-left p-3 rounded-xl border text-xs transition-all duration-200 cursor-pointer ${
                      selectedCategory === cat.id
                        ? "bg-blue-600/15 border-blue-500/50 text-white shadow-[0_0_12px_rgba(0,144,255,0.2)]"
                        : "bg-slate-950/30 border-white/[0.05] text-slate-400 hover:border-white/10 hover:text-slate-200"
                    }`}
                  >
                    <div className="flex items-center gap-2 font-semibold text-sm">
                      <span>{cat.icon}</span>
                      <span className={selectedCategory === cat.id ? "text-white" : "text-slate-300"}>
                        {cat.name}
                      </span>
                    </div>
                    <div className="text-[11px] text-slate-500 mt-1 ml-6">{cat.desc}</div>
                  </button>
                ))}
              </div>
            </div>
          </ScrollReveal>

          {/* Pipeline tuning */}
          <ScrollReveal direction="left" delay={0.1}>
            <div className="rounded-2xl p-5 bg-slate-900/55 border border-white/[0.07] backdrop-blur-xl">
              <h3 className="text-[10px] font-bold uppercase tracking-widest text-slate-400 mb-5 flex items-center gap-2">
                <Sliders className="w-3.5 h-3.5 text-violet-400" />
                AOI Pipeline Parameters
              </h3>

              {/* Blur Kernel */}
              <div className="mb-5">
                <div className="flex justify-between text-xs font-medium mb-2">
                  <span className="text-slate-400">Gaussian Filter Kernel</span>
                  <span className="font-mono text-blue-400">{blurKernel} × {blurKernel}</span>
                </div>
                <input type="range" min="3" max="15" step="2" value={blurKernel}
                  onChange={(e) => setBlurKernel(Number(e.target.value))}
                  className="w-full cursor-pointer accent-blue-500" />
              </div>

              {/* CLAHE Toggle */}
              <div className="flex items-center justify-between py-3 border-y border-white/[0.05] mb-5">
                <div>
                  <div className="text-xs font-semibold text-slate-200">Adaptive Histogram (CLAHE)</div>
                  <div className="text-[11px] text-slate-500 mt-0.5">Amplifies hairline cracks & micro-pits</div>
                </div>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input type="checkbox" checked={useClahe} onChange={(e) => setUseClahe(e.target.checked)} className="sr-only peer" />
                  <div className="w-9 h-5 bg-slate-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-0.5 after:left-0.5 after:bg-white after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-blue-600" />
                </label>
              </div>

              {/* Sensitivity */}
              <div className="mb-5">
                <div className="flex justify-between text-xs font-medium mb-2">
                  <span className="text-slate-400">Segmentation Sensitivity</span>
                  <span className="font-mono text-cyan-400">{sensitivity}%</span>
                </div>
                <input type="range" min="30" max="95" value={sensitivity}
                  onChange={(e) => setSensitivity(Number(e.target.value))}
                  className="w-full cursor-pointer accent-cyan-500" />
              </div>

              {/* Min Area */}
              <div>
                <div className="flex justify-between text-xs font-medium mb-2">
                  <span className="text-slate-400">Noise Suppression (Min Area)</span>
                  <span className="font-mono text-violet-400">{minArea} px</span>
                </div>
                <input type="range" min="50" max="500" step="25" value={minArea}
                  onChange={(e) => setMinArea(Number(e.target.value))}
                  className="w-full cursor-pointer accent-violet-500" />
              </div>
            </div>
          </ScrollReveal>
        </div>

        {/* Inspection display */}
        <div className="lg:col-span-8 flex flex-col gap-5">
          {/* Dual canvas */}
          <ScrollReveal direction="right">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {/* Raw input */}
              <div className="relative rounded-2xl overflow-hidden bg-slate-950 border border-white/[0.07] p-3 shadow-xl">
                <div className="text-[10px] font-mono text-slate-500 mb-2 flex items-center justify-between">
                  <span>INPUT: STEEL_SCRATCH_01.JPG</span>
                  <span className="text-emerald-400">512×512 RGB</span>
                </div>
                <div className="relative aspect-square w-full rounded-xl overflow-hidden bg-slate-900 border border-white/[0.05]">
                  {/* Steel grain texture */}
                  <div className="w-full h-full relative">
                    <div className="absolute inset-0 bg-slate-800" />
                    <div className="absolute inset-0 opacity-50 bg-[repeating-linear-gradient(45deg,#475569_0px,#475569_1px,#1e293b_2px,#1e293b_5px)]" />
                    <div className="absolute inset-0 opacity-20 bg-[repeating-linear-gradient(-30deg,#334155_0px,#334155_1px,transparent_2px,transparent_8px)]" />
                    {/* Scratch mark */}
                    <div className="absolute top-1/2 left-[10%] right-[10%] h-[3px] bg-white/70 rotate-[-20deg] shadow-[0_0_8px_rgba(255,255,255,0.5)] rounded-full blur-[0.5px]" />
                    <div className="absolute top-[52%] left-[15%] right-[20%] h-[1.5px] bg-white/40 rotate-[-22deg] rounded-full" />
                  </div>
                </div>
              </div>

              {/* Processed overlay */}
              <div className="relative rounded-2xl overflow-hidden bg-slate-950 border border-blue-500/25 p-3 shadow-xl">
                <BorderBeam size={220} duration={8} colorFrom="#0090ff" colorTo="#8b5cf6" />
                <div className="text-[10px] font-mono text-blue-400 mb-2 flex items-center justify-between">
                  <span>DEFECT OVERLAY</span>
                  <span className="text-red-400 flex items-center gap-1 font-bold">
                    <AlertTriangle className="w-3 h-3" />
                    DEFECT DETECTED
                  </span>
                </div>
                <div className="relative aspect-square w-full rounded-xl overflow-hidden bg-slate-900 border border-white/[0.05]">
                  <div className="w-full h-full relative">
                    <div className="absolute inset-0 bg-slate-800 opacity-60" />
                    <div className="absolute inset-0 opacity-20 bg-[repeating-linear-gradient(45deg,#475569_0px,#475569_1px,#1e293b_2px,#1e293b_5px)]" />
                    {/* Red highlight mask */}
                    <div className="absolute top-1/2 left-[8%] right-[8%] h-4 bg-red-500/70 rotate-[-20deg] shadow-[0_0_20px_rgba(239,68,68,0.8)] rounded-full border border-red-400/60" />
                    {/* Bounding box */}
                    <div className="absolute top-[38%] left-[8%] right-[10%] h-[28%] border-2 border-red-400/80 bg-red-500/8 rounded-lg flex items-start justify-end p-1.5">
                      <span className="text-[9px] font-mono bg-red-600 text-white px-1.5 py-0.5 rounded font-bold tracking-wider">
                        FRACTURE · 98.4%
                      </span>
                    </div>
                    {/* Crosshair */}
                    <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-5 h-5">
                      <div className="absolute top-1/2 left-0 right-0 h-px bg-cyan-400/50" />
                      <div className="absolute left-1/2 top-0 bottom-0 w-px bg-cyan-400/50" />
                    </div>
                    {/* Scanning beam */}
                    {isScanning && (
                      <motion.div
                        initial={{ top: "0%" }}
                        animate={{ top: "100%" }}
                        transition={{ duration: 1.4, ease: "linear" }}
                        className="absolute left-0 right-0 h-[2px] bg-gradient-to-r from-transparent via-cyan-400 to-transparent shadow-[0_0_15px_#06b6d4]"
                      />
                    )}
                  </div>
                </div>
              </div>
            </div>
          </ScrollReveal>

          {/* Results table */}
          <ScrollReveal direction="up" delay={0.1}>
            <div className="rounded-2xl p-5 bg-slate-900/55 border border-white/[0.07] backdrop-blur-xl">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-[10px] font-bold uppercase tracking-widest text-slate-400 flex items-center gap-2">
                  <FileText className="w-3.5 h-3.5 text-blue-400" />
                  Detected Anomaly Segments
                  <span className="ml-1 px-1.5 py-0.5 rounded bg-blue-500/15 text-blue-400 font-mono text-[9px]">
                    {sampleRegions.length} REGIONS
                  </span>
                </h3>
                <span className="text-[10px] text-slate-500 font-mono hidden sm:block">
                  SCAN #{String(scanCount).padStart(4, "0")}
                </span>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs font-mono">
                  <thead>
                    <tr className="border-b border-white/[0.06] text-[10px] text-slate-500 uppercase tracking-widest">
                      <th className="py-2.5 px-3">ID</th>
                      <th className="py-2.5 px-3">Classification</th>
                      <th className="py-2.5 px-3">Area</th>
                      <th className="py-2.5 px-3">Confidence</th>
                      <th className="py-2.5 px-3 hidden sm:table-cell">Bounding Box</th>
                      <th className="py-2.5 px-3">Severity</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/[0.04]">
                    {sampleRegions.map((r) => (
                      <tr key={r.id} className="hover:bg-white/[0.02] transition-colors">
                        <td className="py-3 px-3 font-bold text-slate-300">#{String(r.id).padStart(3, "0")}</td>
                        <td className="py-3 px-3 font-sans font-medium text-slate-200">{r.type}</td>
                        <td className="py-3 px-3 text-cyan-400">{r.area}</td>
                        <td className="py-3 px-3 text-emerald-400 font-bold">{r.conf}</td>
                        <td className="py-3 px-3 text-slate-500 hidden sm:table-cell">{r.bbox}</td>
                        <td className="py-3 px-3">
                          <span className={`px-2 py-0.5 rounded text-[10px] font-bold tracking-wider ${severityStyle[r.severity]}`}>
                            {r.severity}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </ScrollReveal>
        </div>
      </div>
    </div>
  );
};
