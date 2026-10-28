import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ShieldCheck, Activity, Menu, X, Zap } from "lucide-react";

export const Navbar = ({ activeTab, setActiveTab }) => {
  const [scrolled, setScrolled] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);

  const tabs = [
    { id: "overview",  label: "Platform" },
    { id: "inspect",   label: "Inspector" },
    { id: "pipeline",  label: "Pipeline" },
    { id: "metrics",   label: "Analytics" },
    { id: "export",    label: "Export" },
  ];

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 10);
    window.addEventListener("scroll", handleScroll, { passive: true });
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const handleTabChange = (id) => {
    setActiveTab(id);
    setMobileOpen(false);
  };

  return (
    <>
      <header
        className={`sticky top-0 z-50 w-full px-4 sm:px-8 py-3 transition-all duration-300 ${
          scrolled
            ? "backdrop-blur-2xl bg-[#030712]/90 border-b border-white/[0.06] shadow-[0_1px_30px_rgba(0,0,0,0.5)]"
            : "backdrop-blur-xl bg-[#030712]/70 border-b border-white/[0.04]"
        }`}
      >
        <div className="max-w-7xl mx-auto flex items-center justify-between gap-4">
          {/* Brand */}
          <button
            onClick={() => handleTabChange("overview")}
            className="flex items-center gap-3 group cursor-pointer shrink-0"
          >
            <div className="relative w-9 h-9 rounded-xl bg-gradient-to-br from-blue-500/20 to-violet-500/20 border border-blue-500/30 flex items-center justify-center shadow-[0_0_20px_rgba(0,144,255,0.2)] group-hover:shadow-[0_0_30px_rgba(0,144,255,0.4)] transition-shadow">
              <span className="text-lg select-none">🔬</span>
              <span className="absolute -top-0.5 -right-0.5 w-2.5 h-2.5 rounded-full bg-emerald-400 border-2 border-[#030712] animate-pulse" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-base font-extrabold tracking-tight text-white leading-none">
                  ZeroDefect
                </span>
                <span className="hidden sm:inline text-[9px] font-bold px-1.5 py-0.5 rounded-full bg-blue-500/15 border border-blue-500/30 text-blue-400 tracking-widest uppercase">
                  AOI Suite
                </span>
              </div>
              <div className="text-[10px] font-medium text-slate-500 tracking-widest mt-0.5 hidden sm:block">
                OPTICAL SURFACE INSPECTION
              </div>
            </div>
          </button>

          {/* Desktop Nav */}
          <nav className="hidden md:flex items-center p-1 rounded-full bg-slate-900/80 border border-white/[0.07] shadow-2xl backdrop-blur-md gap-0.5">
            {tabs.map((tab) => {
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => handleTabChange(tab.id)}
                  className={`relative px-4 py-1.5 rounded-full text-xs font-semibold tracking-wide transition-colors cursor-pointer select-none ${
                    isActive ? "text-white" : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  {isActive && (
                    <motion.div
                      layoutId="active-pill"
                      transition={{ type: "spring", stiffness: 400, damping: 32 }}
                      className="absolute inset-0 rounded-full bg-gradient-to-r from-blue-600/40 via-violet-600/35 to-cyan-600/30 border border-blue-400/40 shadow-[0_0_20px_rgba(0,144,255,0.35)]"
                    />
                  )}
                  <span className="relative z-10">{tab.label}</span>
                </button>
              );
            })}
          </nav>

          <div className="flex items-center gap-3">
            {/* Status indicator */}
            <div className="hidden lg:flex items-center gap-2 text-[10px] font-mono text-slate-400 bg-slate-900/60 border border-white/[0.06] px-3 py-1.5 rounded-lg">
              <Activity className="w-3 h-3 text-emerald-400 animate-pulse" />
              <span>SYSTEM READY</span>
            </div>

            {/* Mobile menu toggle */}
            <button
              onClick={() => setMobileOpen(!mobileOpen)}
              className="md:hidden w-8 h-8 rounded-lg bg-slate-900/80 border border-white/10 flex items-center justify-center text-slate-400 hover:text-white transition-colors cursor-pointer"
              aria-label="Toggle menu"
            >
              <AnimatePresence mode="wait">
                {mobileOpen ? (
                  <motion.span key="x" initial={{ rotate: -90, opacity: 0 }} animate={{ rotate: 0, opacity: 1 }} exit={{ rotate: 90, opacity: 0 }} transition={{ duration: 0.15 }}>
                    <X className="w-4 h-4" />
                  </motion.span>
                ) : (
                  <motion.span key="menu" initial={{ rotate: 90, opacity: 0 }} animate={{ rotate: 0, opacity: 1 }} exit={{ rotate: -90, opacity: 0 }} transition={{ duration: 0.15 }}>
                    <Menu className="w-4 h-4" />
                  </motion.span>
                )}
              </AnimatePresence>
            </button>
          </div>
        </div>
      </header>

      {/* Mobile Menu */}
      <AnimatePresence>
        {mobileOpen && (
          <motion.div
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.2, ease: "easeOut" }}
            className="fixed top-[60px] left-0 right-0 z-40 md:hidden bg-[#030712]/95 backdrop-blur-2xl border-b border-white/[0.06] shadow-2xl"
          >
            <nav className="flex flex-col p-4 gap-1 max-w-7xl mx-auto">
              {tabs.map((tab) => {
                const isActive = activeTab === tab.id;
                return (
                  <button
                    key={tab.id}
                    onClick={() => handleTabChange(tab.id)}
                    className={`w-full text-left px-4 py-3 rounded-xl text-sm font-semibold transition-all cursor-pointer ${
                      isActive
                        ? "bg-blue-600/20 border border-blue-500/40 text-white shadow-[0_0_15px_rgba(0,144,255,0.2)]"
                        : "text-slate-400 hover:text-white hover:bg-white/[0.04] border border-transparent"
                    }`}
                  >
                    {tab.label}
                  </button>
                );
              })}
            </nav>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
};
