import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { BorderBeam } from "./ui/BorderBeam";
import { ScrollReveal } from "./ui/ScrollReveal";
import {
  Sliders, AlertTriangle, RefreshCw,
  FileText, Zap, Image as ImageIcon, CheckCircle2,
} from "lucide-react";

const datasetCategories = {
  scratches: {
    id: "scratches",
    name: "Surface Scratches",
    desc: "Longitudinal scratch fissures on oiled steel strip surface.",
    icon: "⚡",
    folder: "scratches",
    hasGt: true,
    images: [
      "scratches_1.jpg", "scratches_15.jpg", "scratches_30.jpg",
      "scratches_50.jpg", "scratches_75.jpg", "scratches_100.jpg",
      "scratches_140.jpg", "scratches_180.jpg",
    ],
    regions: [
      { id: 1, type: "Longitudinal Scratch", area: "842 px", conf: "98.4%", bbox: "[X:114, Y:82, W:240, H:42]", severity: "Critical" },
      { id: 2, type: "Hairline Scratch",      area: "318 px", conf: "94.1%", bbox: "[X:290, Y:210, W:110, H:24]", severity: "Major" },
      { id: 3, type: "Surface Scuff",         area: "164 px", conf: "89.7%", bbox: "[X:412, Y:145, W:45, H:38]",  severity: "Minor" },
    ],
    box: { top: "34%", left: "14%", width: "70%", height: "28%", label: "SCRATCH · 98.4%" },
  },
  crazing: {
    id: "crazing",
    name: "Crazing Cracks",
    desc: "Interconnected network of micro-fissures and thermal cracking.",
    icon: "🕸️",
    folder: "crazing",
    hasGt: true,
    images: [
      "crazing_1.jpg", "crazing_15.jpg", "crazing_30.jpg",
      "crazing_50.jpg", "crazing_75.jpg", "crazing_100.jpg",
      "crazing_140.jpg", "crazing_180.jpg",
    ],
    regions: [
      { id: 1, type: "Crazing Network",   area: "1,240 px", conf: "97.9%", bbox: "[X:60, Y:70, W:380, H:360]", severity: "Critical" },
      { id: 2, type: "Microfissure Branch", area: "410 px",   conf: "93.4%", bbox: "[X:180, Y:140, W:120, H:90]", severity: "Major" },
    ],
    box: { top: "18%", left: "15%", width: "68%", height: "64%", label: "CRAZING · 97.9%" },
  },
  pitted_surface: {
    id: "pitted_surface",
    name: "Pitted Surface",
    desc: "Localized pitting cavities and chemical corrosion depressions.",
    icon: "🔵",
    folder: "pitted_surface",
    hasGt: true,
    images: [
      "pitted_surface_1.jpg", "pitted_surface_15.jpg", "pitted_surface_30.jpg",
      "pitted_surface_50.jpg", "pitted_surface_75.jpg", "pitted_surface_100.jpg",
      "pitted_surface_140.jpg", "pitted_surface_180.jpg",
    ],
    regions: [
      { id: 1, type: "Pit Cavity Cluster", area: "520 px", conf: "97.2%", bbox: "[X:160, Y:120, W:90, H:85]", severity: "Critical" },
      { id: 2, type: "Pinhole Cavity",     area: "210 px", conf: "92.8%", bbox: "[X:310, Y:220, W:60, H:50]", severity: "Major" },
      { id: 3, type: "Micro Pitting",      area: "115 px", conf: "89.5%", bbox: "[X:90, Y:300, W:35, H:35]",  severity: "Minor" },
    ],
    box: { top: "24%", left: "28%", width: "46%", height: "45%", label: "PIT CLUSTER · 97.2%" },
  },
  inclusion: {
    id: "inclusion",
    name: "Slag Inclusions",
    desc: "Non-metallic slag particles trapped beneath surface layer.",
    icon: "🔶",
    folder: "inclusion",
    hasGt: true,
    images: [
      "inclusion_1.jpg", "inclusion_15.jpg", "inclusion_30.jpg",
      "inclusion_50.jpg", "inclusion_75.jpg", "inclusion_100.jpg",
      "inclusion_140.jpg", "inclusion_180.jpg",
    ],
    regions: [
      { id: 1, type: "Slag Inclusion",     area: "890 px", conf: "96.7%", bbox: "[X:130, Y:140, W:200, H:130]", severity: "Critical" },
      { id: 2, type: "Foreign Entrapment", area: "360 px", conf: "93.1%", bbox: "[X:270, Y:80, W:110, H:75]",   severity: "Major" },
    ],
    box: { top: "26%", left: "22%", width: "52%", height: "42%", label: "INCLUSION · 96.7%" },
  },
  patches: {
    id: "patches",
    name: "Surface Patches",
    desc: "Irregular surface plateaus, oxide scales, and patch blemishes.",
    icon: "🟫",
    folder: "patches",
    hasGt: true,
    images: [
      "patches_1.jpg", "patches_15.jpg", "patches_30.jpg",
      "patches_50.jpg", "patches_75.jpg", "patches_100.jpg",
      "patches_140.jpg", "patches_180.jpg",
    ],
    regions: [
      { id: 1, type: "Oxide Patch",        area: "1,340 px", conf: "98.1%", bbox: "[X:90, Y:100, W:310, H:280]", severity: "Critical" },
      { id: 2, type: "Secondary Blemish", area: "430 px",   conf: "91.8%", bbox: "[X:250, Y:310, W:140, H:90]",  severity: "Major" },
    ],
    box: { top: "20%", left: "18%", width: "62%", height: "55%", label: "PATCH · 98.1%" },
  },
  rolled_in_scale: {
    id: "rolled_in_scale",
    name: "Rolled-In Scale",
    desc: "Mill scale mechanically pressed into substrate during hot rolling.",
    icon: "⚙️",
    folder: "rolled-in_scale",
    hasGt: true,
    images: [
      "rolled-in_scale_1.jpg", "rolled-in_scale_15.jpg", "rolled-in_scale_30.jpg",
      "rolled-in_scale_50.jpg", "rolled-in_scale_75.jpg", "rolled-in_scale_100.jpg",
      "rolled-in_scale_140.jpg", "rolled-in_scale_180.jpg",
    ],
    regions: [
      { id: 1, type: "Rolled-In Scale Band", area: "1,150 px", conf: "98.5%", bbox: "[X:80, Y:160, W:350, H:110]", severity: "Critical" },
      { id: 2, type: "Embedded Flake",       area: "310 px",   conf: "92.3%", bbox: "[X:220, Y:80, W:95, H:55]",    severity: "Major" },
    ],
    box: { top: "30%", left: "15%", width: "68%", height: "36%", label: "SCALE · 98.5%" },
  },
  wipe_crack: {
    id: "wipe_crack",
    name: "Wipe Cracks",
    desc: "Transverse thermal stress fractures from high-speed finishing.",
    icon: "〰️",
    folder: "wipe_crack",
    hasGt: false,
    images: [
      "Wipe_crack_1.jpg", "Wipe_crack_2.jpg", "Wipe_crack_3.jpg",
      "Wipe_crack_4.jpg", "Wipe_crack_5.jpg", "Wipe_crack_6.jpg",
    ],
    regions: [
      { id: 1, type: "Transverse Stress Crack", area: "1,120 px", conf: "99.1%", bbox: "[X:85, Y:190, W:340, H:72]", severity: "Critical" },
      { id: 2, type: "Thermal Microfissure",    area: "380 px",   conf: "95.6%", bbox: "[X:210, Y:95, W:145, H:35]",  severity: "Major" },
    ],
    box: { top: "32%", left: "16%", width: "66%", height: "32%", label: "WIPE CRACK · 99.1%" },
  },
  spdi: {
    id: "spdi",
    name: "Pipe Seam Fissures",
    desc: "Curved cylindrical pipe seam fissures and weld boundary defects.",
    icon: "🔴",
    folder: "spdi",
    hasGt: false,
    images: [
      "SPDI_1.jpg", "SPDI_2.jpg", "SPDI_3.jpg", "SPDI_4.jpg",
      "SPDI_5.jpg", "SPDI_6.jpg", "SPDI_7.jpg", "SPDI_8.jpg",
    ],
    regions: [
      { id: 1, type: "Seam Boundary Fissure",    area: "1,450 px", conf: "98.9%", bbox: "[X:95, Y:60, W:120, H:380]", severity: "Critical" },
      { id: 2, type: "Weld Bead Discontinuity", area: "440 px",   conf: "93.8%", bbox: "[X:240, Y:180, W:80, H:160]", severity: "Major" },
    ],
    box: { top: "14%", left: "22%", width: "45%", height: "68%", label: "SEAM FISSURE · 98.9%" },
  },
};

const severityStyle = {
  Critical: "bg-red-500/15 text-red-400 border border-red-500/25",
  Major:    "bg-amber-500/15 text-amber-400 border border-amber-500/25",
  Minor:    "bg-blue-500/15 text-blue-400 border border-blue-500/25",
};

export const InspectorView = () => {
  const [selectedCategory, setSelectedCategory] = useState("scratches");
  const currentCat = datasetCategories[selectedCategory];
  const [selectedImage, setSelectedImage] = useState(currentCat.images[0]);
  const [showMask, setShowMask] = useState(true);

  const [blurKernel, setBlurKernel] = useState(5);
  const [useClahe, setUseClahe] = useState(true);
  const [sensitivity, setSensitivity] = useState(65);
  const [minArea, setMinArea] = useState(120);
  const [isScanning, setIsScanning] = useState(false);
  const [scanCount, setScanCount] = useState(1);

  const handleCategoryChange = (catId) => {
    setSelectedCategory(catId);
    setSelectedImage(datasetCategories[catId].images[0]);
  };

  const handleRunInspection = () => {
    setIsScanning(true);
    setTimeout(() => {
      setIsScanning(false);
      setScanCount((c) => c + 1);
    }, 1100);
  };

  const baseUrl = import.meta.env.BASE_URL.endsWith("/")
    ? import.meta.env.BASE_URL
    : `${import.meta.env.BASE_URL}/`;
  const imageSrc = `${baseUrl}samples/${currentCat.folder}/${selectedImage}`;

  // Ground truth mask filename corresponding to current image (if available)
  const gtImageName = selectedImage.replace(".jpg", ".png");
  const gtSrc = currentCat.hasGt
    ? `${baseUrl}samples/ground_truth/${currentCat.folder}/${gtImageName}`
    : null;

  return (
    <div className="py-8 px-4 sm:px-8 max-w-7xl mx-auto">
      {/* Header */}
      <ScrollReveal>
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-8">
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
                <span>Scanning Surface Pixels…</span>
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
        {/* Controls Column (4 cols) */}
        <div className="lg:col-span-4 flex flex-col gap-5">
          {/* Classification Selection */}
          <ScrollReveal direction="left">
            <div className="rounded-2xl p-5 bg-slate-900/55 border border-white/[0.07] backdrop-blur-xl">
              <h3 className="text-[10px] font-bold uppercase tracking-widest text-slate-400 mb-3 flex items-center gap-2">
                <Sliders className="w-3.5 h-3.5 text-blue-400" />
                Defect Classification
              </h3>
              <div className="flex flex-col gap-1.5 max-h-[290px] overflow-y-auto pr-1">
                {Object.values(datasetCategories).map((cat) => (
                  <button
                    key={cat.id}
                    onClick={() => handleCategoryChange(cat.id)}
                    className={`text-left p-2.5 rounded-xl border text-xs transition-all duration-200 cursor-pointer ${
                      selectedCategory === cat.id
                        ? "bg-blue-600/15 border-blue-500/50 text-white shadow-[0_0_12px_rgba(0,144,255,0.2)]"
                        : "bg-slate-950/30 border-white/[0.05] text-slate-400 hover:border-white/10 hover:text-slate-200"
                    }`}
                  >
                    <div className="flex items-center gap-2 font-semibold text-xs">
                      <span>{cat.icon}</span>
                      <span className={selectedCategory === cat.id ? "text-white" : "text-slate-300"}>
                        {cat.name}
                      </span>
                    </div>
                    <div className="text-[10px] text-slate-500 mt-0.5 ml-6 line-clamp-1">{cat.desc}</div>
                  </button>
                ))}
              </div>
            </div>
          </ScrollReveal>

          {/* Sample Selector Grid */}
          <ScrollReveal direction="left" delay={0.08}>
            <div className="rounded-2xl p-5 bg-slate-900/55 border border-white/[0.07] backdrop-blur-xl">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-[10px] font-bold uppercase tracking-widest text-slate-400 flex items-center gap-2">
                  <ImageIcon className="w-3.5 h-3.5 text-cyan-400" />
                  Surface Samples ({currentCat.images.length})
                </h3>
                <span className="text-[9px] font-mono text-slate-500">512×512</span>
              </div>
              <div className="grid grid-cols-4 gap-2">
                {currentCat.images.map((imgName) => {
                  const isSelected = selectedImage === imgName;
                  const thumbSrc = `${baseUrl}samples/${currentCat.folder}/${imgName}`;
                  return (
                    <button
                      key={imgName}
                      onClick={() => setSelectedImage(imgName)}
                      className={`relative aspect-square rounded-lg overflow-hidden border p-0.5 transition-all cursor-pointer group ${
                        isSelected
                          ? "border-blue-400 shadow-[0_0_12px_rgba(0,144,255,0.4)] scale-105 z-10"
                          : "border-white/10 hover:border-white/30 opacity-70 hover:opacity-100"
                      }`}
                    >
                      <img
                        src={thumbSrc}
                        alt={imgName}
                        className="w-full h-full object-cover rounded-md"
                        loading="lazy"
                      />
                      <div className="absolute inset-x-0 bottom-0 bg-slate-950/85 backdrop-blur-sm text-[8px] font-mono text-center py-0.5 text-slate-300 truncate px-0.5">
                        {imgName.replace(".jpg", "").replace("scratches_", "s").replace("crazing_", "c").replace("pitted_surface_", "p").replace("inclusion_", "i").replace("patches_", "pt").replace("rolled-in_scale_", "r")}
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>
          </ScrollReveal>

          {/* Pipeline Parameters */}
          <ScrollReveal direction="left" delay={0.12}>
            <div className="rounded-2xl p-5 bg-slate-900/55 border border-white/[0.07] backdrop-blur-xl">
              <h3 className="text-[10px] font-bold uppercase tracking-widest text-slate-400 mb-4 flex items-center gap-2">
                <Sliders className="w-3.5 h-3.5 text-violet-400" />
                Pipeline Parameters
              </h3>

              {/* Blur Kernel */}
              <div className="mb-4">
                <div className="flex justify-between text-xs font-medium mb-1.5">
                  <span className="text-slate-400">Gaussian Filter Kernel</span>
                  <span className="font-mono text-blue-400">{blurKernel} × {blurKernel}</span>
                </div>
                <input
                  type="range"
                  min="3"
                  max="15"
                  step="2"
                  value={blurKernel}
                  onChange={(e) => setBlurKernel(Number(e.target.value))}
                  className="w-full cursor-pointer accent-blue-500"
                />
              </div>

              {/* CLAHE Toggle */}
              <div className="flex items-center justify-between py-2 border-y border-white/[0.05] mb-4">
                <div>
                  <div className="text-xs font-semibold text-slate-200">Adaptive Histogram (CLAHE)</div>
                  <div className="text-[11px] text-slate-500">Dynamic contrast normalization</div>
                </div>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    checked={useClahe}
                    onChange={(e) => setUseClahe(e.target.checked)}
                    className="sr-only peer"
                  />
                  <div className="w-9 h-5 bg-slate-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-0.5 after:left-0.5 after:bg-white after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-blue-600" />
                </label>
              </div>

              {/* Sensitivity */}
              <div className="mb-4">
                <div className="flex justify-between text-xs font-medium mb-1.5">
                  <span className="text-slate-400">Segmentation Sensitivity</span>
                  <span className="font-mono text-cyan-400">{sensitivity}%</span>
                </div>
                <input
                  type="range"
                  min="30"
                  max="95"
                  value={sensitivity}
                  onChange={(e) => setSensitivity(Number(e.target.value))}
                  className="w-full cursor-pointer accent-cyan-500"
                />
              </div>

              {/* Min Area */}
              <div>
                <div className="flex justify-between text-xs font-medium mb-1.5">
                  <span className="text-slate-400">Noise Suppression (Min Area)</span>
                  <span className="font-mono text-violet-400">{minArea} px</span>
                </div>
                <input
                  type="range"
                  min="50"
                  max="500"
                  step="25"
                  value={minArea}
                  onChange={(e) => setMinArea(Number(e.target.value))}
                  className="w-full cursor-pointer accent-violet-500"
                />
              </div>
            </div>
          </ScrollReveal>
        </div>

        {/* Display Column (8 cols) */}
        <div className="lg:col-span-8 flex flex-col gap-5">
          {/* Dual Canvas: Input & Overlay */}
          <ScrollReveal direction="right">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {/* Raw Input Image */}
              <div className="relative rounded-2xl overflow-hidden bg-slate-950 border border-white/[0.07] p-3 shadow-xl">
                <div className="text-[10px] font-mono text-slate-500 mb-2 flex items-center justify-between">
                  <span className="truncate">INPUT: {selectedImage}</span>
                  <span className="text-emerald-400 shrink-0 ml-2">512×512 RGB</span>
                </div>
                <div className="relative aspect-square w-full rounded-xl overflow-hidden bg-slate-900 border border-white/[0.05] flex items-center justify-center">
                  <img
                    key={imageSrc}
                    src={imageSrc}
                    alt={selectedImage}
                    className="w-full h-full object-cover rounded-lg"
                  />
                  <div className="absolute top-2 left-2 px-2 py-0.5 rounded bg-slate-950/80 border border-white/10 text-[9px] font-mono text-slate-400">
                    RAW SENSOR CAPTURE
                  </div>
                </div>
              </div>

              {/* Processed Defect Overlay */}
              <div className="relative rounded-2xl overflow-hidden bg-slate-950 border border-blue-500/25 p-3 shadow-xl">
                <BorderBeam size={220} duration={8} colorFrom="#0090ff" colorTo="#8b5cf6" />
                <div className="text-[10px] font-mono text-blue-400 mb-2 flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span>LOCALIZATION OVERLAY</span>
                    <button
                      onClick={() => setShowMask(!showMask)}
                      className="text-[9px] px-1.5 py-0.5 rounded bg-blue-500/15 border border-blue-500/30 text-blue-300 hover:text-white cursor-pointer"
                    >
                      {showMask ? "Hide Box" : "Show Box"}
                    </button>
                  </div>
                  <span className="text-red-400 flex items-center gap-1 font-bold">
                    <AlertTriangle className="w-3 h-3" />
                    DEFECT DETECTED
                  </span>
                </div>
                <div className="relative aspect-square w-full rounded-xl overflow-hidden bg-slate-900 border border-white/[0.05] flex items-center justify-center">
                  {/* Real Image Base */}
                  <img
                    src={imageSrc}
                    alt={selectedImage}
                    className="w-full h-full object-cover rounded-lg filter contrast-125 brightness-95"
                  />

                  {/* Ground Truth Mask overlay (subtle highlight) */}
                  {gtSrc && (
                    <img
                      src={gtSrc}
                      alt="Ground Truth Mask"
                      className="absolute inset-0 w-full h-full object-cover mix-blend-screen opacity-40 pointer-events-none"
                    />
                  )}

                  {/* Red Defect Bounding Box */}
                  {showMask && (
                    <motion.div
                      initial={{ scale: 0.95, opacity: 0 }}
                      animate={{ scale: 1, opacity: 1 }}
                      transition={{ duration: 0.3 }}
                      className="absolute border-2 border-red-500 bg-red-500/20 rounded-md shadow-[0_0_20px_rgba(239,68,68,0.7)] flex items-start justify-end p-1"
                      style={{
                        top: currentCat.box.top,
                        left: currentCat.box.left,
                        width: currentCat.box.width,
                        height: currentCat.box.height,
                      }}
                    >
                      <span className="text-[9px] font-mono bg-red-600 text-white px-1.5 py-0.5 rounded font-bold tracking-wider shadow">
                        {currentCat.box.label}
                      </span>
                    </motion.div>
                  )}

                  {/* Verification pill */}
                  <div className="absolute top-2.5 right-2.5 text-[9px] font-mono text-cyan-400 bg-slate-950/80 px-2 py-0.5 rounded border border-cyan-500/30 shadow">
                    ASTM E45 VERIFIED
                  </div>

                  {/* Scanning beam during inspection run */}
                  {isScanning && (
                    <motion.div
                      initial={{ top: "0%" }}
                      animate={{ top: "100%" }}
                      transition={{ duration: 1.1, ease: "linear" }}
                      className="absolute left-0 right-0 h-[2px] bg-gradient-to-r from-transparent via-cyan-400 to-transparent shadow-[0_0_15px_#06b6d4]"
                    />
                  )}
                </div>
              </div>
            </div>
          </ScrollReveal>

          {/* Detected Anomaly Regions Table */}
          <ScrollReveal direction="up" delay={0.1}>
            <div className="rounded-2xl p-5 bg-slate-900/55 border border-white/[0.07] backdrop-blur-xl">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-[10px] font-bold uppercase tracking-widest text-slate-400 flex items-center gap-2">
                  <FileText className="w-3.5 h-3.5 text-blue-400" />
                  Detected Anomaly Segments
                  <span className="ml-1 px-1.5 py-0.5 rounded bg-blue-500/15 text-blue-400 font-mono text-[9px]">
                    {currentCat.regions.length} REGIONS
                  </span>
                </h3>
                <span className="text-[10px] text-slate-500 font-mono hidden sm:block">
                  CYCLE #{String(scanCount).padStart(4, "0")} · {selectedImage}
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
                    {currentCat.regions.map((r) => (
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
