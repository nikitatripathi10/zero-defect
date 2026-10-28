import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { AuroraBackground } from "./components/ui/AuroraBackground";
import { InteractiveParticles } from "./components/ui/InteractiveParticles";
import { Navbar } from "./components/Navbar";
import { Hero } from "./components/Hero";
import { InspectorView } from "./components/InspectorView";
import { PipelineView } from "./components/PipelineView";
import { MetricsView } from "./components/MetricsView";
import { ExportView } from "./components/ExportView";
import { Footer } from "./components/Footer";

const pageVariants = {
  initial: { opacity: 0, y: 18, filter: "blur(4px)" },
  animate: { opacity: 1, y: 0,  filter: "blur(0px)" },
  exit:    { opacity: 0, y: -14, filter: "blur(2px)" },
};

const pageTransition = { duration: 0.4, ease: [0.16, 1, 0.3, 1] };

export default function App() {
  const [activeTab, setActiveTab] = useState("overview");

  return (
    <AuroraBackground>
      {/* Cursor-reactive particles */}
      <InteractiveParticles quantity={70} />

      {/* Navigation */}
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} />

      {/* Page Content */}
      <main className="relative z-10 min-h-[calc(100vh-180px)]">
        <AnimatePresence mode="wait">
          {activeTab === "overview" && (
            <motion.div
              key="overview"
              variants={pageVariants}
              initial="initial"
              animate="animate"
              exit="exit"
              transition={pageTransition}
            >
              <Hero onExplore={() => setActiveTab("inspect")} />
            </motion.div>
          )}

          {activeTab === "inspect" && (
            <motion.div
              key="inspect"
              variants={pageVariants}
              initial="initial"
              animate="animate"
              exit="exit"
              transition={pageTransition}
            >
              <InspectorView />
            </motion.div>
          )}

          {activeTab === "pipeline" && (
            <motion.div
              key="pipeline"
              variants={pageVariants}
              initial="initial"
              animate="animate"
              exit="exit"
              transition={pageTransition}
            >
              <PipelineView />
            </motion.div>
          )}

          {activeTab === "metrics" && (
            <motion.div
              key="metrics"
              variants={pageVariants}
              initial="initial"
              animate="animate"
              exit="exit"
              transition={pageTransition}
            >
              <MetricsView />
            </motion.div>
          )}

          {activeTab === "export" && (
            <motion.div
              key="export"
              variants={pageVariants}
              initial="initial"
              animate="animate"
              exit="exit"
              transition={pageTransition}
            >
              <ExportView />
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      <Footer />
    </AuroraBackground>
  );
}
