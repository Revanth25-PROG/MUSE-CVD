import axios from 'axios';

const getApiBase = () => {
  if (import.meta.env.VITE_API_URL) {
    const rawUrl = import.meta.env.VITE_API_URL.trim().replace(/\/$/, '');
    return rawUrl.endsWith('/api') ? rawUrl : `${rawUrl}/api`;
  }
  return '/api';
};

const API_BASE = getApiBase();

const api = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json'
  },
  timeout: 30000 // Extended timeout for Render free tier sleep wake-up
});

// Intercept HTML fallback responses (when Vercel SPA rewrites /api/* to index.html before proxy connects)
api.interceptors.response.use((response) => {
  if (typeof response.data === 'string' && (response.data.trim().startsWith('<!doctype') || response.data.trim().startsWith('<html'))) {
    return Promise.reject(new Error("API returned HTML instead of JSON"));
  }
  return response;
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
