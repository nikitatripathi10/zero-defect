import React from "react";
import { ExternalLink, Shield, Zap } from "lucide-react";

export const Footer = () => {
  const year = new Date().getFullYear();

  const links = [
    { label: "Platform Overview", href: "#" },
    { label: "Documentation", href: "#" },
    { label: "Cloud Instance", href: "https://zero-defect-hrjadjlnb3brcphx27pmzh.streamlit.app/", external: true },
    { label: "GitHub", href: "https://github.com/nikitatripathi10/zero-defect", external: true },
  ];

  const capabilities = [
    "Canny Edge Detection",
    "GLCM Texture Analysis",
    "Morphological Segmentation",
    "ASTM Tolerance Reporting",
  ];

  return (
    <footer className="w-full border-t border-white/[0.06] bg-[#030712]/90 backdrop-blur-xl py-12 px-4 sm:px-8 mt-20">
      <div className="max-w-7xl mx-auto">
        {/* Top row */}
        <div className="flex flex-col md:flex-row items-start justify-between gap-10 mb-10">
          {/* Brand */}
          <div className="max-w-xs">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-9 h-9 rounded-xl bg-blue-500/10 border border-blue-500/25 flex items-center justify-center text-lg">
                🔬
              </div>
              <div>
                <div className="text-sm font-extrabold text-white tracking-tight flex items-center gap-2">
                  ZeroDefect
                  <span className="text-[9px] font-bold px-1.5 py-0.5 rounded bg-emerald-500/15 text-emerald-400 font-mono tracking-widest">
                    ENTERPRISE
                  </span>
                </div>
                <div className="text-[10px] text-slate-500 tracking-widest mt-0.5 font-medium">
                  SURFACE INTELLIGENCE PLATFORM
                </div>
              </div>
            </div>
            <p className="text-xs text-slate-500 leading-relaxed">
              Automated Optical Inspection platform delivering precision surface defect detection,
              morphological segmentation, and quality assurance telemetry.
            </p>
          </div>

          {/* Links */}
          <div className="flex flex-wrap gap-x-12 gap-y-6">
            <div>
              <div className="text-[10px] font-bold uppercase tracking-widest text-slate-500 mb-3">Platform</div>
              <ul className="space-y-2">
                {links.map((l) => (
                  <li key={l.label}>
                    <a
                      href={l.href}
                      target={l.external ? "_blank" : undefined}
                      rel={l.external ? "noreferrer" : undefined}
                      className="text-xs text-slate-400 hover:text-white transition-colors flex items-center gap-1 group"
                    >
                      {l.label}
                      {l.external && <ExternalLink className="w-2.5 h-2.5 opacity-0 group-hover:opacity-60 transition-opacity" />}
                    </a>
                  </li>
                ))}
              </ul>
            </div>

            <div>
              <div className="text-[10px] font-bold uppercase tracking-widest text-slate-500 mb-3">Capabilities</div>
              <ul className="space-y-2">
                {capabilities.map((cap) => (
                  <li key={cap} className="text-xs text-slate-400">{cap}</li>
                ))}
              </ul>
            </div>
          </div>
        </div>

        {/* Divider */}
        <div className="h-px bg-gradient-to-r from-transparent via-white/[0.07] to-transparent mb-6" />

        {/* Bottom row */}
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="text-[11px] text-slate-500 font-mono">
            © {year} ZeroDefect Enterprise. All rights reserved.
          </div>
          <div className="flex items-center gap-4 text-[10px] text-slate-600">
            <span className="flex items-center gap-1.5">
              <Zap className="w-3 h-3 text-blue-500" />
              Built with React & Framer Motion
            </span>
            <span>·</span>
            <span className="flex items-center gap-1.5">
              <Shield className="w-3 h-3 text-emerald-500" />
              ISO 9001 Aligned
            </span>
          </div>
        </div>
      </div>
    </footer>
  );
};
