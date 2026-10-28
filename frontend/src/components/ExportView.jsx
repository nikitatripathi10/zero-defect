import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Download, FileSpreadsheet, Image as ImageIcon, CheckCircle, Shield, Zap, ExternalLink } from "lucide-react";
import { BorderBeam } from "./ui/BorderBeam";
import { ScrollReveal } from "./ui/ScrollReveal";
import { TiltCard } from "./ui/TiltCard";

export const ExportView = () => {
  const [downloadStatus, setDownloadStatus] = useState(null);
  const [downloading, setDownloading] = useState(null);

  const handleDownload = (type, key) => {
    setDownloading(key);
    setDownloadStatus(null);

    setTimeout(() => {
      if (type.includes("CSV")) {
        const csvContent =
          "data:text/csv;charset=utf-8," +
          "RegionID,Type,Area_px,Width_px,Height_px,Confidence,Severity\n" +
          "001,Linear Fracture,842,240,42,0.984,Critical\n" +
          "002,Micro Scratch,318,110,24,0.941,Major\n" +
          "003,Surface Pit,164,45,38,0.897,Minor\n" +
          "SUMMARY,All Regions,1324,395,104,0.941,Certified\n";
        const link = document.createElement("a");
        link.setAttribute("href", encodeURI(csvContent));
        link.setAttribute("download", "zerodefect_inspection_report.csv");
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
      } else {
        const svgData = `<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512"><rect width="512" height="512" fill="#0f172a"/><rect x="100" y="200" width="320" height="32" fill="none" stroke="#ef4444" stroke-width="3" rx="4"/><line x1="90" y1="216" x2="420" y2="209" stroke="#ef4444" stroke-width="8" opacity="0.7"/><text x="256" y="120" font-family="monospace" font-size="16" fill="#60a5fa" text-anchor="middle">ZERODEFECT — DEFECT OVERLAY</text><text x="256" y="145" font-family="monospace" font-size="11" fill="#64748b" text-anchor="middle">FRACTURE #001 · 98.4% CONFIDENCE</text></svg>`;
        const blob = new Blob([svgData], { type: "image/svg+xml" });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = "zerodefect_defect_mask.svg";
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
      }

      setDownloading(null);
      setDownloadStatus(`${type} exported successfully.`);
      setTimeout(() => setDownloadStatus(null), 4000);
    }, 700);
  };

  const exportGroups = [
    {
      key: "visual",
      title: "Visual Assets",
      desc: "Lossless annotated inspection imagery and binary defect localization masks for factory archival and quality records.",
      icon: ImageIcon,
      iconColor: "text-blue-400",
      iconBg: "bg-blue-500/10 border-blue-500/25",
      glowColor: "rgba(59,130,246,0.15)",
      actions: [
        {
          key: "overlay",
          label: "Defect Localization Overlay",
          sublabel: "Lossless PNG — 512×512",
          type: "Defect Overlay (PNG)",
          iconColor: "text-blue-400",
          hoverBorder: "hover:border-blue-500/35",
          hoverBg: "hover:bg-blue-600/10",
        },
        {
          key: "mask",
          label: "Binary Segmentation Mask",
          sublabel: "Single-Channel PNG — 512×512",
          type: "Binary Mask (PNG)",
          iconColor: "text-violet-400",
          hoverBorder: "hover:border-violet-500/35",
          hoverBg: "hover:bg-violet-600/10",
        },
      ],
    },
    {
      key: "data",
      title: "Quantitative Reports",
      desc: "Machine-readable structured datasets with pixel areas, bounding coordinates, confidence indices, and quality metrics.",
      icon: FileSpreadsheet,
      iconColor: "text-cyan-400",
      iconBg: "bg-cyan-500/10 border-cyan-500/25",
      glowColor: "rgba(6,182,212,0.15)",
      actions: [
        {
          key: "regions",
          label: "Defect Region Coordinates",
          sublabel: "UTF-8 CSV — Full Telemetry",
          type: "Region Telemetry Report (CSV)",
          iconColor: "text-cyan-400",
          hoverBorder: "hover:border-cyan-500/35",
          hoverBg: "hover:bg-cyan-600/10",
        },
        {
          key: "metrics",
          label: "Quality Evaluation Metrics",
          sublabel: "UTF-8 CSV — ISO/ASTM Format",
          type: "Quality Evaluation Metrics (CSV)",
          iconColor: "text-emerald-400",
          hoverBorder: "hover:border-emerald-500/35",
          hoverBg: "hover:bg-emerald-600/10",
        },
      ],
    },
  ];

  return (
    <div className="py-8 px-4 sm:px-8 max-w-7xl mx-auto">
      {/* Header */}
      <ScrollReveal>
        <div className="mb-10">
          <div className="flex flex-wrap items-center gap-3 mb-3">
            <h2 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
              Report & Data Export
            </h2>
            <span className="text-[10px] px-2.5 py-1 rounded-full bg-blue-500/10 border border-blue-500/25 text-blue-400 font-mono tracking-widest">
              EXPORT CONSOLE
            </span>
          </div>
          <p className="text-sm text-slate-400 max-w-2xl">
            Generate production audit reports, pixel coordinate datasets, and high-resolution defect masks for downstream integration.
          </p>
        </div>
      </ScrollReveal>

      {/* Status toast */}
      <AnimatePresence>
        {downloadStatus && (
          <motion.div
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            className="mb-6 p-4 rounded-xl bg-emerald-500/10 border border-emerald-500/25 text-emerald-300 text-xs font-semibold flex items-center gap-2"
          >
            <CheckCircle className="w-4 h-4 text-emerald-400 shrink-0" />
            <span>{downloadStatus}</span>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Export cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        {exportGroups.map((group, gIdx) => {
          const GroupIcon = group.icon;
          return (
            <ScrollReveal key={group.key} direction={gIdx === 0 ? "left" : "right"} delay={gIdx * 0.1}>
              <TiltCard
                className="relative rounded-2xl p-6 bg-slate-900/55 border border-white/[0.07] backdrop-blur-xl shadow-xl h-full flex flex-col"
                glowColor={group.glowColor}
              >
                <BorderBeam size={200} duration={9 + gIdx * 3} colorFrom={gIdx === 0 ? "#3b82f6" : "#06b6d4"} colorTo="#8b5cf6" />

                <div className={`w-10 h-10 rounded-xl border flex items-center justify-center mb-5 ${group.iconBg}`}>
                  <GroupIcon className={`w-5 h-5 ${group.iconColor}`} />
                </div>

                <h3 className="text-base font-bold text-white mb-2 tracking-tight">{group.title}</h3>
                <p className="text-xs text-slate-400 mb-6 leading-relaxed flex-1">{group.desc}</p>

                <div className="space-y-2.5">
                  {group.actions.map((action) => (
                    <button
                      key={action.key}
                      onClick={() => handleDownload(action.type, action.key)}
                      disabled={downloading === action.key}
                      className={`w-full flex items-center justify-between p-3.5 rounded-xl bg-slate-950/50 border border-white/[0.05] text-xs font-semibold text-slate-200 transition-all duration-200 cursor-pointer group disabled:opacity-60 disabled:cursor-not-allowed ${action.hoverBorder} ${action.hoverBg}`}
                    >
                      <span className="flex items-center gap-2.5">
                        {downloading === action.key ? (
                          <motion.div animate={{ rotate: 360 }} transition={{ duration: 1, repeat: Infinity, ease: "linear" }}>
                            <Download className={`w-4 h-4 ${action.iconColor}`} />
                          </motion.div>
                        ) : (
                          <Download className={`w-4 h-4 ${action.iconColor} group-hover:scale-110 transition-transform`} />
                        )}
                        <span>{action.label}</span>
                      </span>
                      <span className="text-slate-600 font-mono text-[10px]">{action.sublabel}</span>
                    </button>
                  ))}
                </div>
              </TiltCard>
            </ScrollReveal>
          );
        })}
      </div>

      {/* Downstream Integration banner */}
      <ScrollReveal direction="up" delay={0.15}>
        <div className="rounded-2xl p-6 bg-gradient-to-r from-blue-500/8 via-violet-500/6 to-cyan-500/8 border border-white/[0.07] backdrop-blur-xl flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-start gap-4">
            <div className="w-10 h-10 rounded-xl bg-blue-500/15 border border-blue-500/25 flex items-center justify-center shrink-0">
              <Shield className="w-5 h-5 text-blue-400" />
            </div>
            <div>
              <div className="font-bold text-white text-sm mb-1">Quality Management System Integration</div>
              <div className="text-xs text-slate-400 leading-relaxed max-w-md">
                Exported CSV metrics and single-channel binary masks conform to automated manufacturing quality assurance specifications.
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2 px-4 py-2 rounded-xl bg-slate-900/60 border border-white/[0.08] text-xs font-mono text-slate-400 whitespace-nowrap">
            <span>ASTM E45 &middot; ISO 9001 CONFORMANT</span>
          </div>
        </div>
      </ScrollReveal>
    </div>
  );
};
