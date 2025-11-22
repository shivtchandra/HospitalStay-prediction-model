import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import MedicalImpactPredictor from './MedicalImpactPredictor';
import './App.css'; 
import LazyImage from "./components/LazyImage";
import RevealSection from "./components/RevealSection";
import FAQSection from "./Faq";

// Landing + App with React Router wiring
// Tailwind CSS is assumed. This file exposes two routes:
// - `/`  -> Landing page
// - `/app` -> MedicalImpactPredictor (full application)

export default function AppRouter() {
  return (
    <Router>
      <div className="min-h-screen bg-black text-white">
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/app" element={<AppWrapper />} />
        </Routes>
      </div>
    </Router>
  );
}


function AppWrapper() {
  // simple wrapper so the app has a back link
  return (
    <div className="min-h-screen bg-black text-white">
      <div className="max-w-6xl mx-auto p-6">
        <Link to="/" className="inline-block mb-4 text-sm text-white/70">← Back to landing</Link>
        <MedicalImpactPredictor />
      </div>
    </div>
  );
}



function Landing() {
  return (
    <div className="relative min-h-screen overflow-hidden">
      {/* Hospital silhouette background (subtle, white, very low opacity) */}
      <svg className="pointer-events-none absolute inset-0 w-full h-full" preserveAspectRatio="xMidYMid slice" style={{opacity: 0.03}} xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1600 900">
        {/* Simple stylized hospital building silhouette */}
        <g fill="#ffffff">
          <rect x="120" y="420" width="350" height="280" rx="12" />
          <rect x="480" y="340" width="420" height="360" rx="12" />
          <rect x="940" y="460" width="380" height="240" rx="12" />
          <rect x="640" y="260" width="70" height="120" />
          <rect x="720" y="200" width="70" height="180" />
          <rect x="800" y="300" width="70" height="110" />
          <rect x="200" y="360" width="40" height="40" rx="4" />
          <rect x="260" y="360" width="40" height="40" rx="4" />
          <rect x="320" y="360" width="40" height="40" rx="4" />
          <circle cx="730" cy="380" r="18" />
          <rect x="500" y="420" width="40" height="40" rx="6" />
        </g>
      </svg>

      {/* Page content */}
      <header className="relative z-10 max-w-7xl mx-auto px-10 py-8 flex items-center justify-between">
        <div className="flex items-center gap-5">
          <div className="w-12 h-12 rounded-md bg-neutral-900/60 border border-white/6 flex items-center justify-center">
            <div className="w-3 h-3 bg-white rounded-sm" />
          </div>
          <div className="text-lg font-semibold tracking-tight">Medical Impact Predictor</div>
        </div>
        <div className="flex items-center gap-4">
          <Link to="/app" className="px-5 py-3 rounded-full bg-white text-black font-semibold shadow">Get Started</Link>
          {/* <button className="px-4 py-2 rounded-full border border-white/10 text-sm">Request Demo</button> */}
        </div>
      </header>

      <main className="relative z-10">
        {/* Hero */}
        <section className="py-36">
          <div className="max-w-5xl mx-auto text-center px-6">
            <div className="text-sm text-white/60 mb-6">AI-powered LOS prediction.</div>
            <h1 className="text-5xl md:text-6xl font-extrabold tracking-tight leading-tight mb-8">
              AI-powered LOS prediction.
              <br />
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-white to-neutral-400">Enhancing hospital resource management with explainable intelligence.</span>
            </h1>
            <p className="text-lg text-white/60 max-w-3xl mx-auto mb-10">Predict length of stay at admission using clinical and demographic data. Improve bed allocation, staffing and operational planning with calibrated, transparent forecasts.</p>
            <div className="flex items-center justify-center gap-6 mt-6">
              <Link to="/app" className="px-8 py-4 rounded-full bg-white text-black font-semibold shadow-lg">Get Started</Link>
              <a href="#features" className="px-7 py-4 rounded-full border border-white/10 text-white/80">See Features</a>
            </div>
          </div>
        </section>

        {/* Features + tiles */}
        <section id="features" className="max-w-7xl mx-auto px-10 py-20">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-20 items-start">
            <div className="space-y-24 pt-6 pr-6">
              <div>
                <h3 className="text-3xl font-bold mb-5">Smarter patient flow.</h3>
                <p className="text-white/60 text-lg leading-relaxed">Predict length of stay at admission with advanced AI leveraging clinical and demographic data. Enable proactive bed allocation and reduce bottlenecks.</p>
              </div>

              <div>
                <h3 className="text-3xl font-bold mb-5">Explainable AI predictions.</h3>
                <p className="text-white/60 text-lg leading-relaxed">Visual SHAP explanations reveal patient-specific risk factors and highlight model transparency for clinical trust and review.</p>
              </div>

              <div>
                <h3 className="text-3xl font-bold mb-5">Robust, adaptive models.</h3>
                <p className="text-white/60 text-lg leading-relaxed">Hybrid dual-model design: Generalist for standard cases, Specialist for high-risk SAPS-II patients. Optimized for accuracy and reliability.</p>
              </div>
            </div>

            <div className="space-y-10">
  {/* Hero right tile: main dashboard / exterior */}
  <RevealSection className="rounded-3xl overflow-hidden h-96 shadow-2xl" delay={0}>
  <LazyImage
    src="/images/hero-dashboard.png"
    alt="App dashboard preview"
    className="w-full h-full object-cover"
    placeholderClass="bg-neutral-900/40"
  />
</RevealSection>

<div className="grid grid-cols-2 gap-8">
  <RevealSection className="rounded-3xl overflow-hidden h-52 shadow-xl" delay={120}>
    <LazyImage
      src="/images/shap-preview.png"
      alt="SHAP explanation preview"
      className="w-full h-full object-cover"
      placeholderClass="bg-neutral-900/40"
    />
  </RevealSection>

  <RevealSection className="rounded-3xl overflow-hidden h-52 shadow-xl" delay={240}>
    <LazyImage
      src="/images/clinician-bedside.jpg"
      alt="Clinician at bedside"
      className="w-full h-full object-cover"
      placeholderClass="bg-neutral-900/40"
    />
  </RevealSection>
</div>

{/* <RevealSection className="rounded-3xl overflow-hidden h-72 shadow-2xl" delay={360}>
  <LazyImage
    src="/images/hospital-exterior.jpg"
    alt="Hospital exterior"
    className="w-full h-full object-cover"
    placeholderClass="bg-neutral-900/40"
  />
</RevealSection> */}

  {/* Two smaller tiles */}
 
</div>

          </div>
        </section>

        {/* Value cards, stats, FAQ and contact (unchanged structure) */}
      {/* === Metrics + Features Section === */}
      <section className="max-w-7xl mx-auto px-10 py-20">
  <div className="grid md:grid-cols-3 gap-10">
    {/* Quantified uncertainty */}
    <RevealSection delay={0}>
      <div className="p-8 bg-neutral-900/40 rounded-2xl overflow-hidden">
        <LazyImage
          src="/images/one.png"
          alt="Data analysis dashboard in healthcare"
          className="h-60 w-full object-cover rounded-lg mb-6"
          placeholderClass="bg-neutral-800"
        />
        <h4 className="font-semibold text-xl mb-3">Quantified uncertainty.</h4>
        <p className="text-white/60 text-base">
          Delivers probability ranges (10th–90th percentile) for honest, actionable LOS forecasts.
        </p>
      </div>
    </RevealSection>

    {/* Business impact */}
    <RevealSection delay={120}>
      <div className="p-8 bg-neutral-900/40 rounded-2xl overflow-hidden">
        <LazyImage
          src="/images/two.png"
          alt="Hospital operations planning"
          className="h-60 w-full object-cover rounded-lg mb-6"
          placeholderClass="bg-neutral-800"
        />
        <h4 className="font-semibold text-xl mb-3">Business impact.</h4>
        <p className="text-white/60 text-base">
          Improves operational planning: staff scheduling, bed management, and cost estimation.
        </p>
      </div>
    </RevealSection>

    {/* Next-gen deployment */}
    <RevealSection delay={240}>
      <div className="p-8 bg-neutral-900/40 rounded-2xl overflow-hidden">
        <LazyImage
          src="/images/gen.png"
          alt="AI deployment in hospital systems"
          className="h-60 w-full object-cover rounded-lg mb-6"
          placeholderClass="bg-neutral-800"
        />
        <h4 className="font-semibold text-xl mb-3">Next-gen deployment.</h4>
        <p className="text-white/60 text-base">
          API enabled, ready for EHR integration and pilot validation in real environments.
        </p>
      </div>
    </RevealSection>
  </div>
</section>

{/* === Stats Section === */}
<section className="py-16">
  <div className="max-w-4xl mx-auto text-center">
    <div className="grid grid-cols-1 md:grid-cols-2 gap-16 items-center">
      <div>
        <div className="text-5xl font-bold">2.3</div>
        <div className="text-base text-white/60 mt-2">MAE in days</div>
      </div>
      <div>
        <div className="text-5xl font-bold">~80%</div>
        <div className="text-base text-white/60 mt-2">Prediction Interval</div>
      </div>
    </div>

    <div className="mt-12 grid grid-cols-4 gap-12 text-white/60 items-center justify-center">
      {[
        { label: "Real-time AI", icon: "https://cdn-icons-png.flaticon.com/512/1828/1828640.png" },
        { label: "Calibrated estimates", icon: "https://cdn-icons-png.flaticon.com/512/869/869869.png" },
        { label: "SAPS-II driven", icon: "https://cdn-icons-png.flaticon.com/512/535/535239.png" },
        { label: "Visual explanations", icon: "https://cdn-icons-png.flaticon.com/512/3135/3135715.png" }
      ].map((item, i) => (
        <div key={i} className="text-center">
          <div className="w-10 h-10 mx-auto mb-3">
            <img src={item.icon} alt={item.label} className="w-full h-full object-contain opacity-80" />
          </div>
          <div className="text-sm">{item.label}</div>
        </div>
      ))}
    </div>
  </div>
</section>

{/* === FAQ Section === */}
{/* === FAQ Section === */}
<FAQSection />


{/* === Contact / Pilot Section === */}
<section className="max-w-7xl mx-auto px-10 py-24">
  <div className="grid grid-cols-1 md:grid-cols-2 gap-12 items-start">
    {/* <div className="pt-6">
      <h3 className="text-3xl font-bold mb-4">Deploy AI for LOS now.</h3>
      <p className="text-white/60 mb-8">Start your pilot today.</p>
      <form className="space-y-6 max-w-md">
        <input className="w-full p-4 rounded-lg bg-neutral-900 border border-white/6" placeholder="Name" />
        <input className="w-full p-4 rounded-lg bg-neutral-900 border border-white/6" placeholder="Email" />
        <textarea className="w-full p-4 rounded-lg bg-neutral-900 border border-white/6 h-44" placeholder="Message" />
        <button className="px-8 py-4 rounded-full bg-white text-black font-semibold hover:bg-neutral-200 transition">
          Submit
        </button>
      </form>
    </div> */}

    <div className="p-8 bg-neutral-900/20 rounded-2xl">
      <img
        src="https://images.unsplash.com/photo-1586773860418-d37222d8fce3?auto=format&fit=crop&w=900&q=80"
        alt="Team collaboration during hospital pilot"
        className="w-full h-56 object-cover rounded-lg mb-6"
      />
      <h4 className="font-semibold mb-3">Pilot checklist</h4>
      <ul className="text-white/60 list-disc list-inside text-base space-y-2">
        <li>Data integration (EHR)</li>
        <li>Site validation & pilot</li>
        <li>Clinician review & rollout</li>
      </ul>
    </div>
  </div>
</section>

        <footer className="max-w-7xl mx-auto px-10 py-10 border-t border-white/5 text-white/50">
          <div className="flex justify-between items-center">
            <div className="flex items-center gap-4">
              <div className="w-8 h-8 bg-neutral-800 rounded-sm" />
              <div>Medical Impact Predictor</div>
            </div>
            <div className="text-sm">© {new Date().getFullYear()}</div>
          </div>
        </footer>
      </main>
    </div>
  );
}
