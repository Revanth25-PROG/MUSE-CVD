import React from 'react';
import { 
  CheckCircle2, ArrowRight, Layers, ShieldCheck, 
  GitBranch, Sparkles, Cpu, Sliders, Brain, BarChart2 
} from 'lucide-react';

export default function Methodology() {
  const stages = [
    {
      id: 1,
      title: "Dataset Validation & Quality Audit",
      status: "completed",
      description: "Safely read cardiovascular_dataset_cleaned.csv, verified 4,327 records and 20 columns, checked for missing values (0) and duplicates (0).",
      details: "Audit script: src/data_validation.py. Verified exact 4-class target balance.",
      badge: "Completed"
    },
    {
      id: 2,
      title: "Exploratory Data Analysis (EDA)",
      status: "completed",
      description: "Generated target distributions, histograms for demographic, clinical, and ECG interval features, boxplots across classes, and correlation heatmaps.",
      details: "Rendered high-resolution scientific plots into reports/figures/.",
      badge: "Completed"
    },
    {
      id: 3,
      title: "Data Leakage & Provenance Investigation",
      status: "completed",
      description: "Calculated one-way ANOVA F-tests and pairwise Pearson correlations. Discovered feature orthogonality (r=0.011) and between-class independence.",
      details: "Identified zero target leakage; mathematically validated the 25-27% baseline accuracy prior.",
      badge: "Completed"
    },
    {
      id: 4,
      title: "Leakage-Free Preprocessing Pipeline",
      status: "completed",
      description: "Engineered strict 3-way stratified partition: 70% Train (3,028), 15% Validation (649), 15% Held-Out Test (650). Fitted ColumnTransformer exclusively on training split.",
      details: "StandardScaler on 18 numerical features + OneHotEncoder on Gender.",
      badge: "Completed"
    },
    {
      id: 5,
      title: "Baseline Model Benchmarking",
      status: "completed",
      description: "Trained and benchmarked 7 algorithms: Dummy Classifier, Logistic Regression, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, and XGBoost.",
      details: "Evaluated using Accuracy, Macro Precision, Recall, Macro F1, and multi-class One-vs-Rest ROC-AUC.",
      badge: "Completed"
    },
    {
      id: 6,
      title: "Feature-Group Comparison",
      status: "completed",
      description: "Divided the 19 features into clinical/laboratory subset (8 features) versus ECG-derived numerical subset (11 features) to measure standalone predictive strength.",
      details: "Clinical only yielded 28.35% (Extra Trees), ECG only yielded 25.73%.",
      badge: "Completed"
    },
    {
      id: 7,
      title: "Feature-Group Fusion",
      status: "completed",
      description: "Implemented early feature-level concatenation across clinical, laboratory, and ECG numerical domains.",
      details: "Evaluated across 4 classifiers; documented lack of raw waveform modality to prevent misleading multimodal claims.",
      badge: "Completed"
    },
    {
      id: 8,
      title: "Missing-Feature Robustness Analysis",
      status: "completed",
      description: "Subjected the model to controlled MCAR missingness (0%, 10%, 20%, 30%) and tested complete feature-group omission with Mean and Median imputation fallbacks.",
      details: "Omitting ECG group dropped Macro F1 to 0.1992 (-22.6%); median imputation preserved baseline stability.",
      badge: "Completed"
    },
    {
      id: 9,
      title: "Probability Calibration & Uncertainty",
      status: "completed",
      description: "Evaluated multiclass Brier score, Expected Calibration Error (ECE), and fitted 3-fold Platt scaling (Sigmoid) and Isotonic calibration.",
      details: "Platt calibration reduced ECE from 0.0166 down to 0.0019 (an 88.5% improvement in probability fidelity).",
      badge: "Completed"
    },
    {
      id: 10,
      title: "Explainable AI via TreeSHAP",
      status: "completed",
      description: "Computed exact Shapley values using TreeExplainer, generating global feature rankings, multi-class summary plots, and live patient local waterfall drivers.",
      details: "Established clear boundary between model decision weighting and physiological etiology.",
      badge: "Completed"
    },
    {
      id: 11,
      title: "Final Held-Out Test Evaluation & Deployment",
      status: "completed",
      description: "Evaluated the calibrated Random Forest ensemble once on the untouched held-out test set (N=650), generating test confusion matrix, ROC curves, and FastAPI REST endpoints.",
      details: "Test Accuracy: 26.62%, Macro F1: 0.2557, ROC-AUC: 0.5045. All 15 automated pytest tests passing.",
      badge: "Completed"
    }
  ];

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      {/* Header */}
      <div>
        <div className="flex items-center gap-2 text-blue-600 text-xs font-bold uppercase tracking-wider mb-1">
          <Layers className="w-4 h-4" />
          <span>Scientific Execution Lifecycle</span>
        </div>
        <h1 className="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">
          Research Pipeline & Experimental Methodology
        </h1>
        <p className="text-sm text-slate-600 mt-1 max-w-3xl">
          Detailed chronological stages of data curation, statistical leakage audits, model selection, calibration, explainability, and held-out validation.
        </p>
      </div>

      {/* Visual Workflow Flowchart */}
      <div className="bg-white p-6 rounded-2xl border border-slate-200/80 shadow-sm space-y-4">
        <h3 className="text-base font-bold text-slate-900">End-to-End Computational Architecture</h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3 text-xs">
          <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200 space-y-1.5">
            <span className="font-bold text-blue-900 block">1. Ingestion & Audit</span>
            <p className="text-slate-500">4,327 CSV records • 19 features • Schema validation • Zero missingness</p>
          </div>
          <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200 space-y-1.5">
            <span className="font-bold text-blue-900 block">2. Partitioning</span>
            <p className="text-slate-500">Stratified 70/15/15 • Standard scaling • One-hot categorical encoding</p>
          </div>
          <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200 space-y-1.5">
            <span className="font-bold text-blue-900 block">3. Multi-Model Benchmarks</span>
            <p className="text-slate-500">7 models • Feature-group fusion • Controlled MCAR robustness experiments</p>
          </div>
          <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200 space-y-1.5">
            <span className="font-bold text-blue-900 block">4. Calibration & SHAP</span>
            <p className="text-slate-500">Platt scaling (ECE 0.0019) • TreeSHAP global & local feature attributions</p>
          </div>
          <div className="p-3.5 rounded-xl bg-blue-50 border border-blue-200 space-y-1.5">
            <span className="font-bold text-blue-900 block">5. Deployment API & UI</span>
            <p className="text-blue-700">FastAPI port 8000 • React dashboard • Automated pytest verification</p>
          </div>
        </div>
      </div>

      {/* Sequential 11-Stage Cards */}
      <div className="space-y-4">
        {stages.map((stage) => (
          <div 
            key={stage.id} 
            className="bg-white p-5 rounded-xl border border-slate-200/80 shadow-sm hover:border-blue-300 transition-all flex flex-col md:flex-row md:items-center justify-between gap-4"
          >
            <div className="flex items-start gap-3.5">
              <div className="w-8 h-8 rounded-full bg-blue-600 text-white font-bold text-xs flex items-center justify-center shrink-0 mt-0.5">
                {stage.id}
              </div>
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <h3 className="text-sm font-bold text-slate-900">{stage.title}</h3>
                  <span className="inline-flex items-center gap-1 text-[10px] font-semibold text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded-full border border-emerald-200">
                    <CheckCircle2 className="w-3 h-3 text-emerald-600" />
                    {stage.badge}
                  </span>
                </div>
                <p className="text-xs text-slate-600 leading-relaxed max-w-4xl">{stage.description}</p>
                <p className="text-[11px] text-slate-400 font-mono">{stage.details}</p>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
