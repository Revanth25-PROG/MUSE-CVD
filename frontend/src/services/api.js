import axios from 'axios';

// Priority 1: Read Vercel production environment variable VITE_API_URL (e.g., https://muse-cvd-backend.onrender.com)
// Priority 2: In local Vite dev server, use the '/api' proxy
// Priority 3: Fallback to direct local backend
const getApiBase = () => {
  if (import.meta.env.VITE_API_URL) {
    const rawUrl = import.meta.env.VITE_API_URL.trim().replace(/\/$/, '');
    return rawUrl.endsWith('/api') ? rawUrl : `${rawUrl}/api`;
  }
  if (typeof window !== 'undefined') {
    if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
      return window.location.port !== '8008' ? '/api' : 'http://127.0.0.1:8008/api';
    }
  }
  return '/api';
};

const API_BASE = getApiBase();

const api = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json'
  },
  timeout: 25000 // Extended timeout to allow Render free tier spindown wake-ups
});

export const getHealth = async () => {
  const res = await api.get('/health');
  return res.data;
};

export const getDatasetSummary = async () => {
  const res = await api.get('/dataset/summary');
  return res.data;
};

export const getDatasetFeatures = async () => {
  const res = await api.get('/dataset/features');
  return res.data;
};

export const getTargetDistribution = async () => {
  const res = await api.get('/dataset/target-distribution');
  return res.data;
};

export const getModelStatus = async () => {
  const res = await api.get('/model/status');
  return res.data;
};

export const getModelMetrics = async () => {
  const res = await api.get('/model/metrics');
  return res.data;
};

export const getModelComparison = async () => {
  const res = await api.get('/model/comparison');
  return res.data;
};

export const getFeatureFusion = async () => {
  const res = await api.get('/model/feature-fusion');
  return res.data;
};

export const getRobustness = async () => {
  const res = await api.get('/model/robustness');
  return res.data;
};

export const getCalibration = async () => {
  const res = await api.get('/model/calibration');
  return res.data;
};

export const getShapGlobal = async () => {
  const res = await api.get('/model/shap-global');
  return res.data;
};

export const predictPatient = async (patientData) => {
  const res = await api.post('/predict', patientData);
  return res.data;
};

export const explainPatient = async (patientData) => {
  const res = await api.post('/explain', patientData);
  return res.data;
};

export default api;
