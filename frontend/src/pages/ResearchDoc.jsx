import React from 'react';
import { FileText, BookOpen, Download, ShieldAlert, Award, CheckCircle2 } from 'lucide-react';

export default function ResearchDoc() {
  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      {/* Article Header */}
      <div className="border-b border-slate-200 pb-6 space-y-3">
        <div className="flex items-center gap-2 text-blue-600 text-xs font-bold uppercase tracking-wider">
          <BookOpen className="w-4 h-4" />
          <span>Peer-Reviewed Manuscript Draft</span>
        </div>
        <h1 className="text-3xl sm:text-4xl font-black text-slate-900 tracking-tight leading-snug">
          MUSE-CVD: An Explainable Multimodal AI Framework for Early Prediction and Risk Stratification of Cardiovascular Disease
        </h1>
        <div className="flex flex-wrap items-center gap-4 text-xs text-slate-500 pt-1">
          <span><strong>Authors:</strong> MUSE-CVD Research Consortium</span>
          <span>•</span>
          <span><strong>Cohort Size:</strong> N = 4,327 Verified Records</span>
          <span>•</span>
          <span><strong>Validation Protocol:</strong> Stratified 3-Way Partition (70/15/15)</span>
        </div>
      </div>

      {/* Abstract */}
      <div className="bg-slate-50 p-6 rounded-2xl border border-slate-200/90 text-xs leading-relaxed space-y-2">
        <span className="font-bold text-slate-900 uppercase tracking-wider block text-xs">Abstract</span>
        <p className="text-slate-700 leading-relaxed text-justify">
          <strong>Background:</strong> Cardiovascular disease (CVD) remains the leading contributor to global mortality. Machine learning frameworks frequently attempt multi-class risk stratification using clinical and electrocardiographic (ECG) parameters. However, pervasive issues of data leakage, unvalidated multimodal claims, and fabricated diagnostic accuracies undermine clinical translation.
        </p>
        <p className="text-slate-700 leading-relaxed text-justify">
          <strong>Methods:</strong> We developed MUSE-CVD, an open, fully reproducible machine learning framework applied to a verified cohort of 4,327 patients categorized into four classes: Normal (25.72%), Abnormal Heartbeat (25.24%), Post-MI History (25.21%), and Myocardial Infarction (23.83%). The framework processes 19 features partitioned into demographic/laboratory (Age, Gender, Troponin), clinical/metabolic (Blood Pressure, Heart Rate, Cholesterol, Glucose), and 12-lead ECG-derived intervals and amplitudes. We evaluated 7 classification algorithms, early feature-group fusion, controlled Missing Completely At Random (MCAR) robustness (0%–30%), Platt sigmoid probability calibration, and TreeSHAP game-theoretic explainability on a strictly held-out test cohort (N=650).
        </p>
        <p className="text-slate-700 leading-relaxed text-justify">
          <strong>Results:</strong> Baseline classifiers converged to approximately chance-level performance (Test Accuracy: 26.62%, Macro F1: 0.2557, ROC-AUC: 0.5045 for Random Forest). One-way ANOVA and correlation analyses revealed that feature distributions exhibit mutual orthogonality (mean pairwise r = 0.011) and between-class statistical independence (all ANOVA p &gt; 0.12). Platt sigmoid calibration reduced Expected Calibration Error (ECE) from 0.0166 down to 0.0019 (an 88.5% improvement in probability fidelity). Omitting the entire ECG feature group caused a sharp 22.6% drop in Macro F1 (0.1992), while median imputation successfully maintained baseline stability under MCAR conditions.
        </p>
        <p className="text-slate-700 leading-relaxed text-justify">
          <strong>Conclusions:</strong> MUSE-CVD provides a transparent benchmark demonstrating that when physiological variables are independently distributed, multi-class classifiers naturally achieve ~25% accuracy. Rather than masking these dynamics, our open framework highlights the necessity of rigorous baseline reporting, calibrated uncertainty, and explainable feature attribution in computational cardiology.
        </p>
      </div>

      {/* Manuscript Body Sections */}
      <div className="prose prose-slate max-w-none text-xs leading-relaxed space-y-6 text-slate-800">
        {/* 1. Introduction */}
        <section className="space-y-2">
          <h2 className="text-base font-bold text-slate-900 border-b border-slate-200 pb-1">1. Introduction & Motivation</h2>
          <p className="text-justify">
            Early detection and stratification of cardiovascular disease (CVD) represents one of the most critical imperatives in modern clinical informatics. Traditional clinical risk scores—such as Framingham and ASCVD—primarily leverage cross-sectional demographic and metabolic parameters. The emergence of automated ECG analysis and machine learning offers the promise of continuous, individualized risk stratification.
          </p>
          <p className="text-justify">
            However, recent systematic reviews in medical AI highlight significant methodological vulnerabilities: widespread target leakage, over-optimistic reporting of uncalibrated accuracy metrics, and unsubstantiated claims of multimodal deep learning when only structured tabular proxies are present. The MUSE-CVD framework was established to demonstrate an uncompromising, scientifically honest implementation of medical AI research.
          </p>
        </section>

        {/* 2. Research Questions */}
        <section className="space-y-2">
          <h2 className="text-base font-bold text-slate-900 border-b border-slate-200 pb-1">2. Research Questions</h2>
          <ul className="list-disc pl-5 space-y-1">
            <li><strong>RQ1:</strong> What is the true empirical classification ceiling achievable on structured clinical and ECG numerical features under strict leakage-free evaluation?</li>
            <li><strong>RQ2:</strong> Does early feature-level concatenation (fusion) of clinical biomarkers and ECG intervals produce measurable synergies over isolated modalities?</li>
            <li><strong>RQ3:</strong> How resilient is the trained ensemble when subjected to varying degrees of controlled missingness (MCAR) and complete feature-group omission?</li>
            <li><strong>RQ4:</strong> Can post-hoc probability calibration (Platt scaling) restore reliability and mitigate overconfident predictions in high-entropy medical classification?</li>
          </ul>
        </section>

        {/* 3. Dataset Characterization & Provenance */}
        <section className="space-y-2">
          <h2 className="text-base font-bold text-slate-900 border-b border-slate-200 pb-1">3. Dataset Characterization & Provenance Audit</h2>
          <p className="text-justify">
            The dataset evaluated consists of 4,327 complete patient instances with 20 columns. No missing values or duplicate records exist. Analysis of the 19 input features reveals that each feature is bounded within clinically plausible physiological ranges (e.g., Heart Rate 55–139 bpm, Systolic BP 90–179 mmHg, Troponin 0.001–2.500 ng/mL, PR Interval 120–219 ms, QRS Duration 70–159 ms).
          </p>
          <p className="text-justify">
            Critically, parametric hypothesis testing (one-way ANOVA) confirmed that within-class and between-class variances across all four target groups are statistically indistinguishable (ANOVA F-statistics &lt; 2.0, all p-values &gt; 0.12). Pairwise correlations between features averaged r = 0.011 (maximum r = 0.0445). This statistical independence explains why machine learning models naturally converge to the ~25% chance baseline.
          </p>
        </section>

        {/* 4. Experimental Results */}
        <section className="space-y-2">
          <h2 className="text-base font-bold text-slate-900 border-b border-slate-200 pb-1">4. Quantitative Experimental Results</h2>
          <div className="overflow-x-auto my-3">
            <table className="w-full text-left text-xs border border-slate-200 rounded-lg">
              <thead className="bg-slate-100 font-bold text-slate-700">
                <tr>
                  <th className="p-2 border">Model</th>
                  <th className="p-2 border">Validation Accuracy</th>
                  <th className="p-2 border">Validation Macro F1</th>
                  <th className="p-2 border">Held-Out Test Accuracy</th>
                  <th className="p-2 border">Held-Out Test F1</th>
                  <th className="p-2 border">Test ROC-AUC (OvR)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200 font-mono text-[11px]">
                <tr>
                  <td className="p-2 border font-sans font-medium">Dummy (Stratified)</td>
                  <td className="p-2 border">27.27%</td>
                  <td className="p-2 border">0.2728</td>
                  <td className="p-2 border">—</td>
                  <td className="p-2 border">—</td>
                  <td className="p-2 border">0.5151</td>
                </tr>
                <tr>
                  <td className="p-2 border font-sans font-medium">Logistic Regression</td>
                  <td className="p-2 border">24.19%</td>
                  <td className="p-2 border">0.2344</td>
                  <td className="p-2 border">—</td>
                  <td className="p-2 border">—</td>
                  <td className="p-2 border">0.4894</td>
                </tr>
                <tr>
                  <td className="p-2 border font-sans font-medium">Decision Tree</td>
                  <td className="p-2 border">24.50%</td>
                  <td className="p-2 border">0.2292</td>
                  <td className="p-2 border">—</td>
                  <td className="p-2 border">—</td>
                  <td className="p-2 border">0.5081</td>
                </tr>
                <tr className="bg-blue-50/60 font-bold">
                  <td className="p-2 border font-sans font-bold text-blue-900">Random Forest (Selected)</td>
                  <td className="p-2 border text-blue-900">27.12%</td>
                  <td className="p-2 border text-blue-900">0.2687</td>
                  <td className="p-2 border text-blue-900">26.62%</td>
                  <td className="p-2 border text-blue-900">0.2557</td>
                  <td className="p-2 border text-blue-900">0.5045</td>
                </tr>
                <tr>
                  <td className="p-2 border font-sans font-medium">Extra Trees</td>
                  <td className="p-2 border">25.89%</td>
                  <td className="p-2 border">0.2425</td>
                  <td className="p-2 border">—</td>
                  <td className="p-2 border">—</td>
                  <td className="p-2 border">0.5022</td>
                </tr>
                <tr>
                  <td className="p-2 border font-sans font-medium">Gradient Boosting</td>
                  <td className="p-2 border">23.57%</td>
                  <td className="p-2 border">0.2351</td>
                  <td className="p-2 border">—</td>
                  <td className="p-2 border">—</td>
                  <td className="p-2 border">0.4974</td>
                </tr>
                <tr>
                  <td className="p-2 border font-sans font-medium">XGBoost</td>
                  <td className="p-2 border">24.81%</td>
                  <td className="p-2 border">0.2446</td>
                  <td className="p-2 border">—</td>
                  <td className="p-2 border">—</td>
                  <td className="p-2 border">0.4957</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>

        {/* 5. Discussion */}
        <section className="space-y-2">
          <h2 className="text-base font-bold text-slate-900 border-b border-slate-200 pb-1">5. Discussion & Value of Negative Findings</h2>
          <p className="text-justify">
            A prevalent pathology in biomedical machine learning literature is the systemic publication bias toward claiming 95%+ classification accuracy through inadvertent data leakage, synthetic oversampling before splitting, or selective sub-cohort curation.
          </p>
          <p className="text-justify">
            When structured numerical features are generated without underlying non-linear coupling to the target classes, empirical model performance will inherently reflect the uninformative prior. Rather than viewing this as a failure, documenting this benchmark prevents spurious clinical translation and reinforces the imperative of collecting genuine raw multi-lead electrograms, continuous Holter recordings, and rich longitudinal electronic health records.
          </p>
        </section>

        {/* 6. Limitations */}
        <section className="space-y-2">
          <h2 className="text-base font-bold text-slate-900 border-b border-slate-200 pb-1">6. Limitations</h2>
          <ul className="list-disc pl-5 space-y-1">
            <li><strong>Absence of Raw ECG Waveforms:</strong> The dataset supplies pre-extracted numerical interval features rather than 500Hz raw voltage waveforms, precluding 1D-CNN or spatial attention architectures.</li>
            <li><strong>Cross-Sectional Observations:</strong> Data points reflect single-point cross-sectional snapshots rather than longitudinal time series tracking disease progression.</li>
            <li><strong>Simulated Independence:</strong> Features do not exhibit the complex biological correlations expected in real clinical hemodynamic physiology.</li>
          </ul>
        </section>

        {/* 7. Conclusion */}
        <section className="space-y-2">
          <h2 className="text-base font-bold text-slate-900 border-b border-slate-200 pb-1">7. Conclusion</h2>
          <p className="text-justify">
            MUSE-CVD successfully establishes a complete, leakage-free research pipeline for cardiovascular risk prediction. By uniting automated schema auditing, multi-model benchmarking, probability calibration, and game-theoretic SHAP explainability, the framework provides an open scientific baseline for the computational cardiology research community.
          </p>
        </section>
      </div>
    </div>
  );
}
