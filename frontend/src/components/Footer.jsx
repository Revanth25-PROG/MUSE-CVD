import React from 'react';
import { ShieldAlert, BookOpen, GitBranch } from 'lucide-react';

export default function Footer({ setActivePage }) {
  return (
    <footer className="bg-slate-900 text-slate-300 border-t border-slate-800 mt-16">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
        {/* Prominent Mandatory Clinical Disclaimer */}
        <div className="bg-amber-950/40 border border-amber-600/40 rounded-xl p-4 mb-8 flex items-start gap-3">
          <ShieldAlert className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
          <div className="text-xs text-amber-200/90 leading-relaxed">
            <span className="font-bold text-amber-300 uppercase tracking-wider block mb-1">Mandatory Research Disclaimer</span>
            “This system is intended for research and educational purposes only. It is not a medical diagnostic device and must not replace professional medical advice or clinical assessment.”
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          <div className="md:col-span-2 space-y-3">
            <div className="flex items-center gap-2">
              <span className="font-black text-lg text-white tracking-tight">MUSE-CVD</span>
              <span className="text-[10px] bg-blue-900 text-blue-200 font-bold px-2 py-0.5 rounded">Explainable AI</span>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed pr-6">
              An Explainable Multimodal AI Framework for Early Prediction and Risk Stratification of Cardiovascular Disease.
              Benchmarking demographic, clinical, laboratory, and ECG-derived physiological parameters.
            </p>
          </div>

          <div>
            <h4 className="text-xs font-bold text-white uppercase tracking-wider mb-3">Core Modules</h4>
            <ul className="space-y-2 text-xs">
              <li>
                <button onClick={() => setActivePage('predict')} className="hover:text-blue-400 transition-colors">Prediction Workspace</button>
              </li>
              <li>
                <button onClick={() => setActivePage('explain')} className="hover:text-blue-400 transition-colors">SHAP Interpretability</button>
              </li>
              <li>
                <button onClick={() => setActivePage('evaluation')} className="hover:text-blue-400 transition-colors">Model Evaluation & ROC</button>
              </li>
              <li>
                <button onClick={() => setActivePage('dataset')} className="hover:text-blue-400 transition-colors">Dataset Diagnostics</button>
              </li>
            </ul>
          </div>

          <div>
            <h4 className="text-xs font-bold text-white uppercase tracking-wider mb-3">Academic Context</h4>
            <ul className="space-y-2 text-xs">
              <li>
                <button onClick={() => setActivePage('methodology')} className="hover:text-blue-400 transition-colors">11-Stage Pipeline</button>
              </li>
              <li>
                <button onClick={() => setActivePage('research')} className="hover:text-blue-400 transition-colors">Research Manuscript</button>
              </li>
              <li>
                <button onClick={() => setActivePage('about')} className="hover:text-blue-400 transition-colors">System Architecture</button>
              </li>
            </ul>
          </div>
        </div>

        <div className="border-t border-slate-800 mt-8 pt-6 flex flex-col sm:flex-row justify-between items-center text-xs text-slate-500">
          <p>© 2026 MUSE-CVD Research Consortium. All scientific findings validated on verified cohort records.</p>
          <p className="mt-2 sm:mt-0 font-mono text-[11px]">Strict Scientific Integrity Protocol Enabled</p>
        </div>
      </div>
    </footer>
  );
}
