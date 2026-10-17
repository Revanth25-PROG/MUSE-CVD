import React, { useState } from 'react';
import Navbar from './components/Navbar';
import Footer from './components/Footer';

// Pages
import HomeDashboard from './pages/HomeDashboard';
import PredictionWorkspace from './pages/PredictionWorkspace';
import ExplainableAI from './pages/ExplainableAI';
import ModelEvaluation from './pages/ModelEvaluation';
import DatasetExplorer from './pages/DatasetExplorer';
import Methodology from './pages/Methodology';
import ResearchDoc from './pages/ResearchDoc';
import AboutDisclaimer from './pages/AboutDisclaimer';

export default function App() {
  const [activePage, setActivePage] = useState('home');

  const renderPage = () => {
    switch (activePage) {
      case 'home':
        return <HomeDashboard setActivePage={setActivePage} />;
      case 'predict':
        return <PredictionWorkspace />;
      case 'explain':
        return <ExplainableAI />;
      case 'evaluation':
        return <ModelEvaluation />;
      case 'dataset':
        return <DatasetExplorer />;
      case 'methodology':
        return <Methodology />;
      case 'research':
        return <ResearchDoc />;
      case 'about':
        return <AboutDisclaimer />;
      default:
        return <HomeDashboard setActivePage={setActivePage} />;
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 text-slate-900 font-sans">
      <Navbar activePage={activePage} setActivePage={setActivePage} />
      <main className="flex-1">
        {renderPage()}
      </main>
      <Footer setActivePage={setActivePage} />
    </div>
  );
}
