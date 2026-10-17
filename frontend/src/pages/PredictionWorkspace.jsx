import React, { useState } from 'react';
import { 
  HeartPulse, Send, RotateCcw, Brain, Activity, AlertTriangle, 
  CheckCircle2, Sparkles, TrendingUp, TrendingDown, Layers, HelpCircle, ShieldAlert 
} from 'lucide-react';
import { 
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell 
} from 'recharts';
import { predictPatient, explainPatient } from '../services/api';

const presets = {
  sampleA: {
    name: 'Patient 001 (Suspected Acute MI Profile)',
    data: {
      Age: 73, Gender: 'Female', Heart_Rate: 134, Systolic_BP: 170, Diastolic_BP: 108,
      Cholesterol: 214, Glucose: 127, Troponin: 1.339, PR_Interval: 217, QRS_Duration: 147,
      QT_Interval: 420, P_Wave_Duration: 96, T_Wave_Duration: 164, ST_Segment_Level: -0.09,
      RR_Interval: 935, HRV_SDNN: 117, HRV_RMSSD: 92, R_Peak_Amplitude: 1.94, S_Wave_Amplitude: -2.27
    }
  },
  sampleB: {
    name: 'Patient 002 (Normal Ambulatory Profile)',
    data: {
      Age: 45, Gender: 'Male', Heart_Rate: 72, Systolic_BP: 118, Diastolic_BP: 78,
      Cholesterol: 185, Glucose: 92, Troponin: 0.012, PR_Interval: 155, QRS_Duration: 90,
      QT_Interval: 380, P_Wave_Duration: 90, T_Wave_Duration: 145, ST_Segment_Level: 0.01,
      RR_Interval: 830, HRV_SDNN: 55, HRV_RMSSD: 42, R_Peak_Amplitude: 1.85, S_Wave_Amplitude: -1.2
    }
  },
  sampleC: {
    name: 'Patient 003 (Post-MI Follow-up Profile)',
    data: {
      Age: 64, Gender: 'Male', Heart_Rate: 80, Systolic_BP: 167, Diastolic_BP: 61,
      Cholesterol: 233, Glucose: 186, Troponin: 1.221, PR_Interval: 145, QRS_Duration: 153,
      QT_Interval: 458, P_Wave_Duration: 104, T_Wave_Duration: 155, ST_Segment_Level: 0.404,
      RR_Interval: 1102, HRV_SDNN: 58, HRV_RMSSD: 73, R_Peak_Amplitude: 1.55, S_Wave_Amplitude: -1.62
    }
  },
  sampleD: {
    name: 'Patient 004 (Arrhythmia Profile)',
    data: {
      Age: 52, Gender: 'Male', Heart_Rate: 89, Systolic_BP: 155, Diastolic_BP: 97,
      Cholesterol: 249, Glucose: 186, Troponin: 2.484, PR_Interval: 151, QRS_Duration: 153,
      QT_Interval: 410, P_Wave_Duration: 118, T_Wave_Duration: 183, ST_Segment_Level: -0.169,
      RR_Interval: 1187, HRV_SDNN: 93, HRV_RMSSD: 41, R_Peak_Amplitude: 2.74, S_Wave_Amplitude: -0.38
    }
  }
};

const defaultValues = presets.sampleA.data;

export default function PredictionWorkspace() {
  const [formData, setFormData] = useState(defaultValues);
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    const { name, value, type } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'number' ? (value === '' ? '' : parseFloat(value)) : value
    }));
  };

  const handlePresetSelect = (presetKey) => {
    setFormData(presets[presetKey].data);
    setPrediction(null);
    setExplanation(null);
    setError(null);
  };

  const handleReset = () => {
    setFormData(defaultValues);
    setPrediction(null);
    setExplanation(null);
    setError(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      // Clean data types for API submission
      const payload = {
        Age: parseInt(formData.Age, 10),
        Gender: formData.Gender,
        Heart_Rate: parseInt(formData.Heart_Rate, 10),
        Systolic_BP: parseInt(formData.Systolic_BP, 10),
        Diastolic_BP: parseInt(formData.Diastolic_BP, 10),
        Cholesterol: parseInt(formData.Cholesterol, 10),
        Glucose: parseInt(formData.Glucose, 10),
        Troponin: parseFloat(formData.Troponin),
        PR_Interval: parseInt(formData.PR_Interval, 10),
        QRS_Duration: parseInt(formData.QRS_Duration, 10),
        QT_Interval: parseInt(formData.QT_Interval, 10),
        P_Wave_Duration: parseInt(formData.P_Wave_Duration, 10),
        T_Wave_Duration: parseInt(formData.T_Wave_Duration, 10),
        ST_Segment_Level: parseFloat(formData.ST_Segment_Level),
        RR_Interval: parseInt(formData.RR_Interval, 10),
        HRV_SDNN: parseInt(formData.HRV_SDNN, 10),
        HRV_RMSSD: parseInt(formData.HRV_RMSSD, 10),
        R_Peak_Amplitude: parseFloat(formData.R_Peak_Amplitude),
        S_Wave_Amplitude: parseFloat(formData.S_Wave_Amplitude)
      };

      const [predResult, expResult] = await Promise.all([
        predictPatient(payload),
        explainPatient(payload)
      ]);

      setPrediction(predResult);
      setExplanation(expResult);
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.detail || "Inference error: verify that backend is running on port 8000.");
    } finally {
      setLoading(false);
    }
  };

  const chartData = prediction ? Object.entries(prediction.class_probabilities).map(([name, prob]) => ({
    name: name.replace('_', ' '),
    rawName: name,
    probability: (prob * 100),
    isPredicted: name === prediction.predicted_class
  })) : [];

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      {/* Header */}
      <div>
        <div className="flex items-center gap-2 text-blue-600 text-xs font-bold uppercase tracking-wider mb-1">
          <HeartPulse className="w-4 h-4" />
          <span>Interactive Clinical Inference</span>
        </div>
        <h1 className="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">
          Cardiovascular Disease Prediction Workspace
        </h1>
        <p className="text-sm text-slate-600 mt-1 max-w-3xl">
          Enter patient measurements to execute live machine learning prediction with probability calibration and real-time SHAP feature attribution.
        </p>
      </div>

      {/* Preset Profiles Bar */}
      <div className="bg-white p-4 rounded-xl border border-slate-200/80 shadow-sm flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
        <div className="flex items-center gap-2 text-xs font-semibold text-slate-700">
          <Sparkles className="w-4 h-4 text-blue-600" />
          <span>Load Standard Clinical Presets:</span>
        </div>
        <div className="flex flex-wrap gap-2">
          {Object.entries(presets).map(([key, p]) => (
            <button
              key={key}
              type="button"
              onClick={() => handlePresetSelect(key)}
              className="px-3 py-1.5 text-xs font-medium rounded-lg bg-slate-100 hover:bg-blue-50 hover:text-blue-700 transition-colors border border-slate-200"
            >
              {p.name.split(' ')[1]} ({p.name.split('(')[1]?.replace(')', '')})
            </button>
          ))}
        </div>
      </div>

      {/* Error Alert */}
      {error && (
        <div className="p-4 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 text-xs flex items-start gap-3">
          <AlertTriangle className="w-5 h-5 text-rose-600 shrink-0" />
          <div>
            <span className="font-bold block mb-0.5">Inference Request Failed</span>
            {error}
          </div>
        </div>
      )}

      {/* Form and Results Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left (7 Cols): Patient Features Input Form */}
        <form onSubmit={handleSubmit} className="lg:col-span-7 space-y-6">
          {/* Group 1: Demographics & Biomarkers */}
          <div className="bg-white p-5 rounded-xl border border-slate-200/80 shadow-sm space-y-4">
            <h3 className="text-sm font-bold text-slate-900 border-b border-slate-100 pb-2 flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-blue-600"></span>
              Demographics & Cardiac Biomarkers
            </h3>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <div>
                <label className="block text-xs font-medium text-slate-700 mb-1">Age (years)</label>
                <input
                  type="number"
                  name="Age"
                  min="18"
                  max="120"
                  required
                  value={formData.Age}
                  onChange={handleChange}
                  className="w-full px-3 py-2 text-xs border border-slate-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:outline-none"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-700 mb-1">Biological Sex</label>
                <select
                  name="Gender"
                  value={formData.Gender}
                  onChange={handleChange}
                  className="w-full px-3 py-2 text-xs border border-slate-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:outline-none bg-white"
                >
                  <option value="Male">Male</option>
                  <option value="Female">Female</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-700 mb-1">Cardiac Troponin (ng/mL)</label>
                <input
                  type="number"
                  step="0.001"
                  min="0"
                  max="10"
                  name="Troponin"
                  required
                  value={formData.Troponin}
                  onChange={handleChange}
                  className="w-full px-3 py-2 text-xs border border-slate-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:outline-none"
                />
              </div>
            </div>
          </div>

          {/* Group 2: Hemodynamic & Metabolic */}
          <div className="bg-white p-5 rounded-xl border border-slate-200/80 shadow-sm space-y-4">
            <h3 className="text-sm font-bold text-slate-900 border-b border-slate-100 pb-2 flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-sky-500"></span>
              Hemodynamics & Metabolism
            </h3>
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
              <div>
                <label className="block text-xs font-medium text-slate-700 mb-1">Heart Rate (bpm)</label>
                <input
                  type="number"
                  name="Heart_Rate"
                  min="30"
                  max="220"
                  required
                  value={formData.Heart_Rate}
                  onChange={handleChange}
                  className="w-full px-3 py-2 text-xs border border-slate-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:outline-none"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-700 mb-1">Systolic BP (mmHg)</label>
                <input
                  type="number"
                  name="Systolic_BP"
                  min="60"
                  max="250"
                  required
                  value={formData.Systolic_BP}
                  onChange={handleChange}
                  className="w-full px-3 py-2 text-xs border border-slate-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:outline-none"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-700 mb-1">Diastolic BP (mmHg)</label>
                <input
                  type="number"
                  name="Diastolic_BP"
                  min="30"
                  max="150"
                  required
                  value={formData.Diastolic_BP}
                  onChange={handleChange}
                  className="w-full px-3 py-2 text-xs border border-slate-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:outline-none"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-700 mb-1">Cholesterol (mg/dL)</label>
                <input
                  type="number"
                  name="Cholesterol"
                  min="50"
                  max="600"
                  required
                  value={formData.Cholesterol}
                  onChange={handleChange}
                  className="w-full px-3 py-2 text-xs border border-slate-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:outline-none"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-700 mb-1">Fasting Glucose (mg/dL)</label>
                <input
                  type="number"
                  name="Glucose"
                  min="30"
                  max="600"
                  required
                  value={formData.Glucose}
                  onChange={handleChange}
                  className="w-full px-3 py-2 text-xs border border-slate-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:outline-none"
                />
              </div>
            </div>
          </div>

          {/* Group 3: 12-Lead ECG Intervals & Morphology */}
          <div className="bg-white p-5 rounded-xl border border-slate-200/80 shadow-sm space-y-4">
            <h3 className="text-sm font-bold text-slate-900 border-b border-slate-100 pb-2 flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-indigo-600"></span>
              ECG Intervals, Durations & Amplitudes
            </h3>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              <div>
                <label className="block text-xs font-medium text-slate-700 mb-1">PR Interval (ms)</label>
                <input
                  type="number"
                  name="PR_Interval"
                  value={formData.PR_Interval}
                  onChange={handleChange}
                  className="w-full px-3 py-2 text-xs border border-slate-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:outline-none"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-700 mb-1">QRS Duration (ms)</label>
                <input
                  type="number"
                  name="QRS_Duration"
                  value={formData.QRS_Duration}
                  onChange={handleChange}
                  className="w-full px-3 py-2 text-xs border border-slate-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:outline-none"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-700 mb-1">QT Interval (ms)</label>
                <input
                  type="number"
                  name="QT_Interval"
                  value={formData.QT_Interval}
                  onChange={handleChange}
                  className="w-full px-3 py-2 text-xs border border-slate-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:outline-none"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-700 mb-1">RR Interval (ms)</label>
                <input
                  type="number"
                  name="RR_Interval"
                  value={formData.RR_Interval}
                  onChange={handleChange}
                  className="w-full px-3 py-2 text-xs border border-slate-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:outline-none"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-700 mb-1">P Wave Dur (ms)</label>
                <input
                  type="number"
                  name="P_Wave_Duration"
                  value={formData.P_Wave_Duration}
                  onChange={handleChange}
                  className="w-full px-3 py-2 text-xs border border-slate-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:outline-none"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-700 mb-1">T Wave Dur (ms)</label>
                <input
                  type="number"
                  name="T_Wave_Duration"
                  value={formData.T_Wave_Duration}
                  onChange={handleChange}
                  className="w-full px-3 py-2 text-xs border border-slate-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:outline-none"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-700 mb-1">ST Level (mV)</label>
                <input
                  type="number"
                  step="0.001"
                  name="ST_Segment_Level"
                  value={formData.ST_Segment_Level}
                  onChange={handleChange}
                  className="w-full px-3 py-2 text-xs border border-slate-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:outline-none"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-700 mb-1">HRV SDNN (ms)</label>
                <input
                  type="number"
                  name="HRV_SDNN"
                  value={formData.HRV_SDNN}
                  onChange={handleChange}
                  className="w-full px-3 py-2 text-xs border border-slate-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:outline-none"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-700 mb-1">HRV RMSSD (ms)</label>
                <input
                  type="number"
                  name="HRV_RMSSD"
                  value={formData.HRV_RMSSD}
                  onChange={handleChange}
                  className="w-full px-3 py-2 text-xs border border-slate-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:outline-none"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-700 mb-1">R Peak Amp (mV)</label>
                <input
                  type="number"
                  step="0.01"
                  name="R_Peak_Amplitude"
                  value={formData.R_Peak_Amplitude}
                  onChange={handleChange}
                  className="w-full px-3 py-2 text-xs border border-slate-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:outline-none"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-700 mb-1">S Wave Amp (mV)</label>
                <input
                  type="number"
                  step="0.01"
                  name="S_Wave_Amplitude"
                  value={formData.S_Wave_Amplitude}
                  onChange={handleChange}
                  className="w-full px-3 py-2 text-xs border border-slate-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:outline-none"
                />
              </div>
            </div>
          </div>

          {/* Form Actions */}
          <div className="flex items-center gap-3">
            <button
              type="submit"
              disabled={loading}
              className="flex-1 inline-flex items-center justify-center gap-2 px-6 py-3 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-bold text-sm shadow-md shadow-blue-600/30 transition-all disabled:opacity-50"
            >
              {loading ? (
                <>
                  <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></span>
                  <span>Executing Pipeline...</span>
                </>
              ) : (
                <>
                  <Send className="w-4 h-4" />
                  <span>Run Model Prediction & Explain</span>
                </>
              )}
            </button>
            <button
              type="button"
              onClick={handleReset}
              className="px-4 py-3 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-700 font-semibold text-sm transition-colors flex items-center gap-1.5"
            >
              <RotateCcw className="w-4 h-4" />
              <span>Reset</span>
            </button>
          </div>
        </form>

        {/* Right (5 Cols): Live Results & SHAP Drivers Panel */}
        <div className="lg:col-span-5 space-y-6">
          {prediction ? (
            <div className="space-y-6 animate-fadeIn">
              {/* Predicted State Badge Card */}
              <div className="bg-white p-6 rounded-2xl border border-slate-200/80 shadow-sm space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Classification Output</span>
                  <span className="text-[10px] bg-slate-100 text-slate-600 font-mono px-2 py-0.5 rounded">Entropy: {prediction.prediction_entropy}</span>
                </div>

                <div className="p-4 rounded-xl bg-gradient-to-r from-blue-50 to-sky-50 border border-blue-200">
                  <div className="text-xs text-blue-700 font-semibold">Predicted Dataset Category:</div>
                  <div className="text-2xl font-black text-blue-950 mt-1">
                    {prediction.predicted_class.replace('_', ' ')}
                  </div>
                  <div className="text-xs text-blue-600 mt-1">
                    Calibrated Confidence: <strong>{(prediction.confidence * 100).toFixed(1)}%</strong>
                  </div>
                </div>

                {/* Probability Distribution Chart */}
                <div>
                  <h4 className="text-xs font-bold text-slate-700 uppercase tracking-wider mb-2">Class Probability Distribution</h4>
                  <div className="h-44 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={chartData} layout="vertical" margin={{ top: 5, right: 30, left: 40, bottom: 5 }}>
                        <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 10 }} unit="%" />
                        <YAxis type="category" dataKey="name" tick={{ fontSize: 10 }} width={100} />
                        <Tooltip formatter={(val) => [`${val.toFixed(1)}%`, 'Calibrated Probability']} />
                        <Bar dataKey="probability" radius={[0, 4, 4, 0]}>
                          {chartData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.isPredicted ? '#2563eb' : '#94a3b8'} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div className="text-[11px] text-slate-400 border-t border-slate-100 pt-3 flex items-start gap-1.5">
                  <ShieldAlert className="w-3.5 h-3.5 text-amber-500 shrink-0 mt-0.5" />
                  <span>{prediction.disclaimer}</span>
                </div>
              </div>

              {/* Local SHAP Explanations */}
              {explanation && (
                <div className="bg-white p-6 rounded-2xl border border-slate-200/80 shadow-sm space-y-4">
                  <div className="flex items-center gap-2 text-slate-900 border-b border-slate-100 pb-3">
                    <Brain className="w-5 h-5 text-blue-600" />
                    <div>
                      <h4 className="text-sm font-bold">Local SHAP Feature Attribution</h4>
                      <p className="text-[11px] text-slate-500">Key physiological parameters driving this specific decision</p>
                    </div>
                  </div>

                  <div className="space-y-2">
                    {explanation.top_drivers.map((drv, i) => {
                      const isPos = drv.direction === 'positive';
                      return (
                        <div key={i} className="flex items-center justify-between p-2.5 rounded-lg bg-slate-50 border border-slate-100 text-xs">
                          <div className="flex items-center gap-2">
                            {isPos ? (
                              <TrendingUp className="w-4 h-4 text-emerald-600" />
                            ) : (
                              <TrendingDown className="w-4 h-4 text-rose-600" />
                            )}
                            <span className="font-semibold text-slate-800">{drv.feature}</span>
                          </div>
                          <div className="flex items-center gap-2">
                            <span className={`font-mono font-bold ${isPos ? 'text-emerald-700' : 'text-rose-700'}`}>
                              {drv.shap_value > 0 ? `+${drv.shap_value}` : drv.shap_value}
                            </span>
                            <span className="text-[10px] text-slate-400 capitalize">({drv.direction})</span>
                          </div>
                        </div>
                      );
                    })}
                  </div>

                  <p className="text-[11px] text-slate-400 italic">
                    Methodology: {explanation.methodology}
                  </p>
                </div>
              )}
            </div>
          ) : (
            <div className="bg-white p-12 rounded-2xl border border-dashed border-slate-300 text-center space-y-4 flex flex-col items-center justify-center min-h-[380px]">
              <div className="w-12 h-12 rounded-full bg-blue-50 text-blue-600 flex items-center justify-center">
                <HeartPulse className="w-6 h-6 animate-pulse" />
              </div>
              <div className="max-w-xs space-y-1">
                <h4 className="text-sm font-bold text-slate-800">Awaiting Patient Measurements</h4>
                <p className="text-xs text-slate-500">
                  Select a clinical preset above or adjust physiological values, then click "Run Model Prediction" to inspect live outcomes.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
