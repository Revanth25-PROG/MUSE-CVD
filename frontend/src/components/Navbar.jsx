import React, { useState, useEffect } from 'react';
import { Activity, ShieldAlert, BarChart2, Database, Brain, Cpu, FileText, Info, CheckCircle2, AlertCircle } from 'lucide-react';
import { getHealth } from '../services/api';

export default function Navbar({ activePage, setActivePage }) {
  const [apiStatus, setApiStatus] = useState('checking');

  useEffect(() => {
    const checkApi = async () => {
      try {
        const data = await getHealth();
        if (data.status === 'healthy') {
          setApiStatus('connected');
        } else {
          setApiStatus('error');
        }
      } catch (err) {
        setApiStatus('disconnected');
      }
    };
    checkApi();
    const interval = setInterval(checkApi, 20000);
    return () => clearInterval(interval);
  }, []);

  const navItems = [
    { id: 'home', label: 'Home Dashboard', icon: Activity },
    { id: 'predict', label: 'Prediction Workspace', icon: Cpu },
    { id: 'explain', label: 'Explainable AI', icon: Brain },
    { id: 'evaluation', label: 'Model Evaluation', icon: BarChart2 },
    { id: 'dataset', label: 'Dataset Explorer', icon: Database },
    { id: 'methodology', label: 'Methodology', icon: Activity },
    { id: 'research', label: 'Research Manuscript', icon: FileText },
    { id: 'about', label: 'About & Disclaimer', icon: Info },
  ];

  return (
    <header className="bg-white border-b border-slate-200 sticky top-0 z-50 shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo and Brand */}
          <div 
            onClick={() => setActivePage('home')}
            className="flex items-center gap-3 cursor-pointer group"
          >
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-blue-700 to-sky-500 flex items-center justify-center text-white shadow-md shadow-blue-500/20 group-hover:scale-105 transition-transform">
              <Activity className="w-6 h-6 animate-pulse" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-extrabold text-xl tracking-tight text-slate-900">MUSE-CVD</span>
                <span className="text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 bg-blue-100 text-blue-800 rounded-full">Research v1.0</span>
              </div>
              <p className="text-xs text-slate-500 hidden sm:block">Explainable Cardiovascular AI Framework</p>
            </div>
          </div>

          {/* System Health Badge */}
          <div className="flex items-center gap-3">
            <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border ${
              apiStatus === 'connected' ? 'bg-emerald-50 text-emerald-700 border-emerald-200' :
              apiStatus === 'checking' ? 'bg-amber-50 text-amber-700 border-amber-200' :
              'bg-rose-50 text-rose-700 border-rose-200'
            }`}>
              {apiStatus === 'connected' ? (
                <>
                  <span className="w-2 h-2 rounded-full bg-emerald-500 animate-ping" />
                  <span className="w-2 h-2 rounded-full bg-emerald-500 -ml-3.5" />
                  <span>API Active (Port 8008)</span>
                </>
              ) : apiStatus === 'checking' ? (
                <>
                  <span className="w-2 h-2 rounded-full bg-amber-500" />
                  <span>Connecting API...</span>
                </>
              ) : (
                <>
                  <AlertCircle className="w-3.5 h-3.5 text-rose-600" />
                  <span>API Offline</span>
                </>
              )}
            </div>
          </div>
        </div>

        {/* Navigation Tabs Bar */}
        <div className="flex space-x-1 overflow-x-auto py-1 scrollbar-none border-t border-slate-100">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activePage === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActivePage(item.id)}
                className={`flex items-center gap-2 px-3.5 py-2 text-xs font-semibold rounded-lg whitespace-nowrap transition-all ${
                  isActive
                    ? 'bg-blue-600 text-white shadow-sm shadow-blue-600/30'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
                }`}
              >
                <Icon className={`w-3.5 h-3.5 ${isActive ? 'text-white' : 'text-slate-400'}`} />
                {item.label}
              </button>
            );
          })}
        </div>
      </div>
    </header>
  );
}
