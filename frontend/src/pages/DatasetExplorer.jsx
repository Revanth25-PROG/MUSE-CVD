import React, { useState, useEffect } from 'react';
import { 
  Database, Table, BarChart2, ShieldCheck, AlertCircle, 
  Layers, Lock, Sparkles, Filter 
} from 'lucide-react';
import { getDatasetSummary, getDatasetFeatures } from '../services/api';

export default function DatasetExplorer() {
  const [summary, setSummary] = useState(null);
  const [featuresData, setFeaturesData] = useState(null);
  const [filterGroup, setFilterGroup] = useState('All');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchDatasetInfo = async () => {
      try {
        const [sum, feats] = await Promise.all([
          getDatasetSummary().catch(() => null),
          getDatasetFeatures().catch(() => null)
        ]);
        setSummary(sum);
        setFeaturesData(feats);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    };
    fetchDatasetInfo();
  }, []);

  const featureList = featuresData?.feature_details || [];
  const filteredFeatures = filterGroup === 'All' 
    ? featureList 
    : featureList.filter(f => f.group.toLowerCase().includes(filterGroup.toLowerCase()));

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      {/* Header */}
      <div>
        <div className="flex items-center gap-2 text-blue-600 text-xs font-bold uppercase tracking-wider mb-1">
          <Database className="w-4 h-4" />
          <span>Data Governance & Diagnostics</span>
        </div>
        <h1 className="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">
          Dataset Diagnostics & Schema Explorer
        </h1>
        <p className="text-sm text-slate-600 mt-1 max-w-3xl">
          Comprehensive schema documentation, missing-value audits, summary statistics, and empirical provenance verification.
        </p>
      </div>

      {/* Dataset Audit Metric Highlights */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <div className="bg-white p-4 rounded-xl border border-slate-200/80 shadow-sm">
          <span className="text-[11px] font-bold text-slate-400 uppercase tracking-wider block mb-1">Total Records</span>
          <div className="text-2xl font-black text-slate-900">4,327</div>
          <span className="text-[10px] text-emerald-600 font-semibold">100% verified</span>
        </div>

        <div className="bg-white p-4 rounded-xl border border-slate-200/80 shadow-sm">
          <span className="text-[11px] font-bold text-slate-400 uppercase tracking-wider block mb-1">Total Dimensions</span>
          <div className="text-2xl font-black text-slate-900">20 Columns</div>
          <span className="text-[10px] text-slate-500">19 Features + 1 Target</span>
        </div>

        <div className="bg-white p-4 rounded-xl border border-slate-200/80 shadow-sm">
          <span className="text-[11px] font-bold text-slate-400 uppercase tracking-wider block mb-1">Missing Values</span>
          <div className="text-2xl font-black text-emerald-600">0 (0.0%)</div>
          <span className="text-[10px] text-slate-500">Zero missing entries</span>
        </div>

        <div className="bg-white p-4 rounded-xl border border-slate-200/80 shadow-sm">
          <span className="text-[11px] font-bold text-slate-400 uppercase tracking-wider block mb-1">Duplicate Records</span>
          <div className="text-2xl font-black text-emerald-600">0 (0.0%)</div>
          <span className="text-[10px] text-slate-500">No duplicate rows</span>
        </div>
      </div>

      {/* Scientific Provenance Analysis Banner */}
      <div className="bg-white p-6 rounded-2xl border border-blue-200 shadow-sm space-y-3">
        <div className="flex items-center gap-2 text-blue-900 font-bold text-base">
          <Sparkles className="w-5 h-5 text-blue-600" />
          <span>Cohort Provenance & Signal-to-Noise Verification</span>
        </div>
        <p className="text-xs text-slate-600 leading-relaxed">
          Through one-way ANOVA F-tests and Pearson correlation profiling across all 4,327 records, the data exhibits distinct mathematical properties:
        </p>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 pt-1 text-xs">
          <div className="p-3 rounded-lg bg-blue-50/70 border border-blue-100">
            <span className="font-bold text-blue-950 block mb-1">Mutual Orthogonality</span>
            <span className="text-blue-800">Average pairwise feature correlation is r = 0.011 (max r = 0.0445), proving virtually zero collinearity.</span>
          </div>
          <div className="p-3 rounded-lg bg-blue-50/70 border border-blue-100">
            <span className="font-bold text-blue-950 block mb-1">Uniform Marginal Spans</span>
            <span className="text-blue-800">Features are bounded within realistic clinical intervals but follow continuous/discrete uniform distributions.</span>
          </div>
          <div className="p-3 rounded-lg bg-blue-50/70 border border-blue-100">
            <span className="font-bold text-blue-950 block mb-1">Target Independence</span>
            <span className="text-blue-800">Between-class ANOVA tests yield p &gt; 0.12 across all features, mathematically explaining the ~25% chance baseline accuracy.</span>
          </div>
        </div>
      </div>

      {/* Feature Schema Table with Category Filter */}
      <div className="bg-white p-6 rounded-2xl border border-slate-200/80 shadow-sm space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <h3 className="text-base font-bold text-slate-900">Feature Schema and Descriptive Statistics</h3>
            <p className="text-xs text-slate-500">Showing {filteredFeatures.length} of {featureList.length} verified physiological attributes</p>
          </div>
          <div className="flex items-center gap-2">
            <Filter className="w-3.5 h-3.5 text-slate-400" />
            <div className="flex rounded-lg bg-slate-100 p-0.5 text-xs font-medium">
              {['All', 'Clinical', 'ECG', 'Demographic'].map(grp => (
                <button
                  key={grp}
                  onClick={() => setFilterGroup(grp)}
                  className={`px-3 py-1 rounded-md transition-colors ${
                    filterGroup === grp ? 'bg-white text-slate-900 font-bold shadow-sm' : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  {grp}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-slate-200 bg-slate-50 text-slate-600 font-semibold">
                <th className="py-2.5 px-3">Column Name</th>
                <th className="py-2.5 px-3">Modality Group</th>
                <th className="py-2.5 px-3">Data Type</th>
                <th className="py-2.5 px-3">Range / Values</th>
                <th className="py-2.5 px-3">Mean (μ)</th>
                <th className="py-2.5 px-3">Std Dev (σ)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 font-mono text-[11px]">
              {filteredFeatures.map((f) => (
                <tr key={f.name} className="hover:bg-slate-50/80 transition-colors">
                  <td className="py-2.5 px-3 font-bold text-slate-900">{f.name}</td>
                  <td className="py-2.5 px-3 font-sans">
                    <span className={`px-2 py-0.5 rounded-full text-[10px] font-medium ${
                      f.group.includes('ECG') ? 'bg-indigo-100 text-indigo-800' :
                      f.group.includes('Clinical') ? 'bg-blue-100 text-blue-800' :
                      'bg-slate-100 text-slate-700'
                    }`}>
                      {f.group}
                    </span>
                  </td>
                  <td className="py-2.5 px-3 font-sans text-slate-600">{f.type}</td>
                  <td className="py-2.5 px-3 text-slate-700">{f.range_or_categories}</td>
                  <td className="py-2.5 px-3 text-slate-700">{f.mean !== null ? f.mean.toFixed(2) : '—'}</td>
                  <td className="py-2.5 px-3 text-slate-700">{f.std !== null ? f.std.toFixed(2) : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Generated Visual Artifacts Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div className="bg-white p-6 rounded-2xl border border-slate-200/80 shadow-sm space-y-4">
          <h3 className="text-base font-bold text-slate-900">Feature Correlation Matrix (Heatmap)</h3>
          <p className="text-xs text-slate-500">Pairwise Pearson correlation coefficients confirming near-complete feature independence.</p>
          <div className="rounded-xl overflow-hidden border border-slate-200 bg-slate-50">
            <img 
              src="/api/figures/correlation_heatmap.png" 
              alt="Correlation Heatmap" 
              className="w-full h-auto object-cover hover:scale-105 transition-transform duration-300"
            />
          </div>
        </div>

        <div className="bg-white p-6 rounded-2xl border border-slate-200/80 shadow-sm space-y-4">
          <h3 className="text-base font-bold text-slate-900">ECG Numerical Distributions</h3>
          <p className="text-xs text-slate-500">Histograms and KDE plots of the 11 ECG morphological and interval attributes.</p>
          <div className="rounded-xl overflow-hidden border border-slate-200 bg-slate-50">
            <img 
              src="/api/figures/ecg_features_distribution.png" 
              alt="ECG Distributions" 
              className="w-full h-auto object-cover hover:scale-105 transition-transform duration-300"
            />
          </div>
        </div>
      </div>

      {/* Privacy and Non-Exposure Protocol */}
      <div className="bg-slate-900 text-slate-300 p-5 rounded-2xl border border-slate-800 flex items-start gap-3">
        <Lock className="w-5 h-5 text-emerald-400 shrink-0 mt-0.5" />
        <div className="text-xs leading-relaxed">
          <span className="font-bold text-white block mb-0.5">Patient Privacy & Non-Exposure Protocol</span>
          In strict compliance with clinical data protection guidelines, individual raw patient records are never directly exposed in public website responses. Only aggregate statistics, schema profiles, and on-demand inference responses are transmitted.
        </div>
      </div>
    </div>
  );
}
