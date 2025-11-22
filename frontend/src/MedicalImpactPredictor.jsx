import React, { useState } from 'react';
import {
  Activity,
  DollarSign,
  Calendar,
  TrendingUp,
  AlertCircle,
  Stethoscope,
  Sparkles,
  ArrowRight,
  CheckCircle,
  Info,
  TrendingDown,
  ChevronUp
} from 'lucide-react';

export default function MedicalImpactPredictor() {
  const [formData, setFormData] = useState({
    anchor_age: 70,
    gender: 'M',
    admission_type: 'EMERGENCY',
    insurance: 'Medicare',
    primary_diagnosis: 'Sepsis',
    procedure_count: 2,
    max_creatinine: 1.8,
    min_hemoglobin: 9.5,
    saps_ii_score: 35,
    avg_heart_rate: 105,
    average_daily_patient_cost: '',
    currency: 'INR'
  });

  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const numberFields = [
    'anchor_age',
    'procedure_count',
    'max_creatinine',
    'min_hemoglobin',
    'saps_ii_score',
    'avg_heart_rate',
    'average_daily_patient_cost'
  ];

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: numberFields.includes(name) ? (value === '' ? '' : parseFloat(value) || 0) : value
    }));
  };

  const handleSubmit = async (e) => {
    if (e && e.preventDefault) e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const resp = await fetch('http://127.0.0.1:5000/predict_impact', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      });
      if (!resp.ok) {
        const errData = await resp.json().catch(() => ({}));
        throw new Error(errData.error || 'Failed to get prediction');
      }
      const data = await resp.json();
      setResults(data);
    } catch (err) {
      setError(err.message || String(err));
    } finally {
      setLoading(false);
    }
  };

  const getRiskBadgeColor = (riskLevel) => {
    if (!riskLevel) return 'bg-gray-600/20 text-gray-300 border-gray-500/30';
    const level = riskLevel.toLowerCase();
    if (level.includes('high')) return 'bg-red-900/40 text-red-300 border-red-700/50';
    if (level.includes('medium')) return 'bg-yellow-900/40 text-yellow-300 border-yellow-700/50';
    return 'bg-green-900/40 text-green-300 border-green-700/50';
  };

  const formatCurrency = (amount, currency) => {
    try {
      const num = parseFloat(amount) || 0;
      return new Intl.NumberFormat('en-IN', {
        style: 'currency',
        currency: currency || 'INR',
        maximumFractionDigits: 0
      }).format(num);
    } catch {
      return String(amount || 0);
    }
  };

  return (
    <div className="min-h-screen relative overflow-x-hidden bg-gradient-to-b from-[#0a0a0a] via-[#141414] to-[#1a1a1a] text-white antialiased">
      <div className="absolute inset-0 opacity-40 pointer-events-none" style={{
        backgroundImage: 'url("data:image/svg+xml,%3Csvg width=\'40\' height=\'40\' xmlns=\'http://www.w3.org/2000/svg\'%3E%3Cdefs%3E%3Cpattern id=\'grid\' width=\'40\' height=\'40\' patternUnits=\'userSpaceOnUse\'%3E%3Cpath d=\'M 0 10 L 40 10 M 10 0 L 10 40 M 0 20 L 40 20 M 20 0 L 20 40 M 0 30 L 40 30 M 30 0 L 30 40\' fill=\'none\' stroke=\'rgba(255,255,255,0.02)\' stroke-width=\'1\'/%3E%3C/pattern%3E%3C/defs%3E%3Crect width=\'100%25\' height=\'100%25\' fill=\'url(%23grid)\'/%3E%3C/svg%3E")'
      }}></div>
      
      <div className="absolute top-20 left-10 w-64 h-64 bg-gradient-to-br from-gray-300/5 to-transparent rounded-full blur-3xl pointer-events-none"></div>
      <div className="absolute bottom-40 right-20 w-96 h-96 bg-gradient-to-tl from-gray-400/5 to-transparent rounded-full blur-3xl pointer-events-none"></div>

      <header className="sticky top-0 z-50 bg-black/80 backdrop-blur-xl border-b border-gray-700/30">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-gray-200 via-gray-300 to-gray-400 flex items-center justify-center shadow-[0_8px_30px_rgba(180,180,180,0.25)]">
              <Stethoscope className="w-6 h-6 text-black" />
            </div>
            <div>
              <div className="text-lg font-semibold bg-gradient-to-r from-white to-gray-300 bg-clip-text text-transparent">Medical Impact Predictor</div>
              <div className="text-sm text-gray-400">AI-powered clinical insights</div>
            </div>
          </div>

          <div className="flex items-center gap-3 bg-gray-800/50 px-4 py-2 rounded-full border border-gray-600/30 backdrop-blur-sm">
            <Sparkles className="w-4 h-4 text-gray-300" />
            <span className="text-sm text-gray-200 font-semibold">Advanced Analytics</span>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-12 lg:py-20 relative z-10">
        <section className="text-center mb-12">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-gradient-to-r from-gray-800/50 to-gray-700/50 border border-gray-600/30 backdrop-blur-sm mb-6">
            <Activity className="w-4 h-4 text-gray-300" />
            <span className="text-sm text-gray-200 font-semibold">Predictive Healthcare Technology</span>
          </div>

          <h1 className="text-4xl md:text-5xl font-extrabold leading-tight">
            Predict Patient Outcomes
            <span className="block mt-2 text-transparent bg-clip-text bg-gradient-to-r from-gray-100 via-gray-300 to-gray-400">with Precision</span>
          </h1>

          <p className="mt-4 text-lg text-gray-400 max-w-2xl mx-auto">
            Leverage machine learning to estimate length of stay and healthcare costs with clinical-grade accuracy.
          </p>
        </section>

        <div className="grid gap-8 lg:grid-cols-[1fr_500px]">
          <div className="bg-gradient-to-br from-gray-900/90 to-gray-800/50 rounded-2xl p-6 md:p-8 shadow-2xl border border-gray-700/50 backdrop-blur-sm">
            <div className="flex items-center gap-3 mb-4">
              <Activity className="w-5 h-5 text-gray-300" />
              <h3 className="font-semibold text-lg m-0 text-gray-100">Patient Information</h3>
            </div>

            <div className="space-y-6">
              <div>
                <div className="text-xs font-semibold uppercase text-gray-400 tracking-wide mb-3">Demographics</div>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-1">Age</label>
                    <input
                      type="number"
                      name="anchor_age"
                      value={formData.anchor_age}
                      onChange={handleInputChange}
                      className="w-full p-3 rounded-lg bg-black/40 border border-gray-600/50 text-white text-sm focus:border-gray-400 focus:ring-1 focus:ring-gray-400 outline-none transition-all"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-1">Gender</label>
                    <select
                      name="gender"
                      value={formData.gender}
                      onChange={handleInputChange}
                      className="w-full p-3 rounded-lg bg-black/40 border border-gray-600/50 text-white text-sm focus:border-gray-400 focus:ring-1 focus:ring-gray-400 outline-none transition-all"
                    >
                      <option value="M">Male</option>
                      <option value="F">Female</option>
                    </select>
                  </div>
                </div>
              </div>

              <div>
                <div className="text-xs font-semibold uppercase text-gray-400 tracking-wide mb-3">Admission Details</div>
                <div className="grid grid-cols-2 gap-3 mb-3">
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-1">Type</label>
                    <select
                      name="admission_type"
                      value={formData.admission_type}
                      onChange={handleInputChange}
                      className="w-full p-3 rounded-lg bg-black/40 border border-gray-600/50 text-white text-sm focus:border-gray-400 focus:ring-1 focus:ring-gray-400 outline-none transition-all"
                    >
                      <option value="EMERGENCY">Emergency</option>
                      <option value="URGENT">Urgent</option>
                      <option value="ELECTIVE">Elective</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-1">Insurance</label>
                    <select
                      name="insurance"
                      value={formData.insurance}
                      onChange={handleInputChange}
                      className="w-full p-3 rounded-lg bg-black/40 border border-gray-600/50 text-white text-sm focus:border-gray-400 focus:ring-1 focus:ring-gray-400 outline-none transition-all"
                    >
                      <option value="Medicare">Medicare</option>
                      <option value="Medicaid">Medicaid</option>
                      <option value="Private">Private</option>
                      <option value="Other">Other</option>
                    </select>
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-1">Primary Diagnosis</label>
                  <input
                    name="primary_diagnosis"
                    value={formData.primary_diagnosis}
                    onChange={handleInputChange}
                    className="w-full p-3 rounded-lg bg-black/40 border border-gray-600/50 text-white text-sm focus:border-gray-400 focus:ring-1 focus:ring-gray-400 outline-none transition-all"
                  />
                </div>
              </div>

              <div>
                <div className="text-xs font-semibold uppercase text-gray-400 tracking-wide mb-3">Clinical Metrics</div>
                <div className="grid grid-cols-3 gap-3">
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-1">Procedures</label>
                    <input
                      name="procedure_count"
                      type="number"
                      value={formData.procedure_count}
                      onChange={handleInputChange}
                      className="w-full p-3 rounded-lg bg-black/40 border border-gray-600/50 text-white text-sm focus:border-gray-400 focus:ring-1 focus:ring-gray-400 outline-none transition-all"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-1">SAPS II</label>
                    <input
                      name="saps_ii_score"
                      type="number"
                      value={formData.saps_ii_score}
                      onChange={handleInputChange}
                      className="w-full p-3 rounded-lg bg-black/40 border border-gray-600/50 text-white text-sm focus:border-gray-400 focus:ring-1 focus:ring-gray-400 outline-none transition-all"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-1">Heart Rate</label>
                    <input
                      name="avg_heart_rate"
                      type="number"
                      value={formData.avg_heart_rate}
                      onChange={handleInputChange}
                      className="w-full p-3 rounded-lg bg-black/40 border border-gray-600/50 text-white text-sm focus:border-gray-400 focus:ring-1 focus:ring-gray-400 outline-none transition-all"
                    />
                  </div>
                </div>
              </div>

              <div>
                <div className="text-xs font-semibold uppercase text-gray-400 tracking-wide mb-3">Lab Values</div>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-1">Max Creatinine</label>
                    <input
                      name="max_creatinine"
                      type="number"
                      step="0.1"
                      value={formData.max_creatinine}
                      onChange={handleInputChange}
                      className="w-full p-3 rounded-lg bg-black/40 border border-gray-600/50 text-white text-sm focus:border-gray-400 focus:ring-1 focus:ring-gray-400 outline-none transition-all"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-1">Min Hemoglobin</label>
                    <input
                      name="min_hemoglobin"
                      type="number"
                      step="0.1"
                      value={formData.min_hemoglobin}
                      onChange={handleInputChange}
                      className="w-full p-3 rounded-lg bg-black/40 border border-gray-600/50 text-white text-sm focus:border-gray-400 focus:ring-1 focus:ring-gray-400 outline-none transition-all"
                    />
                  </div>
                </div>
              </div>

              <div>
                <button
                  type="button"
                  onClick={handleSubmit}
                  disabled={loading}
                  className={`w-full inline-flex items-center justify-center gap-3 py-3 rounded-lg font-bold text-white shadow-lg disabled:opacity-60 transition-all ${
                    loading ? 'bg-gray-600' : 'bg-gradient-to-r from-gray-700 via-gray-600 to-gray-700 hover:from-gray-600 hover:via-gray-500 hover:to-gray-600 hover:shadow-gray-500/30'
                  }`}
                >
                  {loading ? (
                    <>
                      <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin inline-block" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      Generate Prediction <ArrowRight className="w-4 h-4" />
                    </>
                  )}
                </button>
              </div>

              {error && (
                <div className="p-3 rounded-md bg-red-900/20 border border-red-700/40 text-red-200">
                  <strong className="font-semibold">Prediction Failed:</strong> <span>{error}</span>
                </div>
              )}
            </div>
          </div>

          <aside className="flex flex-col gap-5">
            <div className="bg-gradient-to-br from-gray-900/90 to-gray-800/50 rounded-lg p-4 border border-gray-700/50 backdrop-blur-sm">
              <div className="text-xs font-semibold uppercase text-gray-400 tracking-wide mb-3">Cost Parameters</div>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-1">Daily Cost</label>
                  <input
                    name="average_daily_patient_cost"
                    type="number"
                    step="0.01"
                    value={formData.average_daily_patient_cost}
                    onChange={handleInputChange}
                    placeholder="e.g., 2500"
                    className="w-full p-3 rounded-lg bg-black/40 border border-gray-600/50 text-white text-sm focus:border-gray-400 focus:ring-1 focus:ring-gray-400 outline-none transition-all"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-1">Currency</label>
                  <select
                    name="currency"
                    value={formData.currency}
                    onChange={handleInputChange}
                    className="w-full p-3 rounded-lg bg-black/40 border border-gray-600/50 text-white text-sm focus:border-gray-400 focus:ring-1 focus:ring-gray-400 outline-none transition-all"
                  >
                    <option value="INR">INR (₹)</option>
                    <option value="USD">USD ($)</option>
                    <option value="EUR">EUR (€)</option>
                    <option value="GBP">GBP (£)</option>
                  </select>
                </div>
              </div>
            </div>

            {results ? (
              <div className="flex flex-col gap-4">
                <div className="bg-gradient-to-br from-gray-900/90 to-gray-800/50 rounded-lg p-4 border border-gray-700/50 backdrop-blur-sm">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <Info className="w-4 h-4 text-gray-300" />
                      <span className="text-sm font-semibold text-gray-200">Patient Classification</span>
                    </div>
                    <span className={`text-xs px-2 py-1 rounded-full border ${getRiskBadgeColor(results.patient_info?.risk_level)}`}>
                      {results.patient_info?.risk_level || 'Unknown'}
                    </span>
                  </div>
                  
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Model Used:</span>
                      <span className="font-semibold text-gray-200">{results.patient_info?.model_used || results.model_used}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">SAPS-II Score:</span>
                      <span className="font-semibold text-gray-200">{results.patient_info?.saps_ii_score ?? results.saps_ii_score}</span>
                    </div>
                    {results.patient_info?.diagnosis_category && (
                      <div className="flex justify-between">
                        <span className="text-gray-400">Diagnosis Category:</span>
                        <span className="font-semibold text-gray-200 capitalize">{results.patient_info.diagnosis_category}</span>
                      </div>
                    )}
                  </div>
                </div>

                <div className="bg-gradient-to-br from-gray-800/80 via-gray-700/50 to-transparent rounded-lg p-4 border border-gray-600/40 shadow-lg">
                  <div className="flex items-center gap-3 mb-3">
                    <Calendar className="w-5 h-5 text-gray-300" />
                    <div className="text-xs uppercase text-gray-300 font-semibold">Length of Stay</div>
                  </div>

                  <div className="text-3xl font-extrabold bg-gradient-to-r from-white to-gray-300 bg-clip-text text-transparent">
                    {(results.prediction?.los_median_days ?? results.predicted_los_median_days ?? 0).toFixed(1)}
                  </div>
                  <div className="text-sm text-gray-400">days (median)</div>

                  <div className="mt-3 border-t border-gray-600/30 pt-3 space-y-2">
                    <div className="flex justify-between text-sm">
                      <div className="text-gray-400 flex items-center gap-1">
                        <TrendingDown className="w-3 h-3" />
                        Optimistic (P10)
                      </div>
                      <div className="font-semibold text-gray-200">
                        {(results.prediction?.los_p10_days ?? results.predicted_los_p10_days ?? 0).toFixed(1)} days
                      </div>
                    </div>
                    <div className="flex justify-between text-sm">
                      <div className="text-gray-400 flex items-center gap-1">
                        <ChevronUp className="w-3 h-3" />
                        Conservative (P90)
                      </div>
                      <div className="font-semibold text-gray-200">
                        {(results.prediction?.los_p90_days ?? results.predicted_los_p90_days ?? 0).toFixed(1)} days
                      </div>
                    </div>
                  </div>

                  {results.interpretation?.confidence_range && (
                    <div className="mt-3 pt-3 border-t border-gray-600/30">
                      <div className="text-xs text-gray-400">80% Confidence Interval:</div>
                      <div className="text-sm font-semibold text-gray-200">{results.interpretation.confidence_range}</div>
                    </div>
                  )}
                </div>

                <div className="bg-gradient-to-br from-gray-900/90 to-gray-800/50 rounded-lg p-4 border border-gray-700/50 backdrop-blur-sm">
                  <div className="flex items-center gap-3 mb-3">
                    <DollarSign className="w-5 h-5 text-gray-300" />
                    <div className="text-sm font-semibold text-gray-200">Estimated Cost</div>
                  </div>

                  <div className="text-2xl font-extrabold bg-gradient-to-r from-white to-gray-300 bg-clip-text text-transparent">
                    {results.cost_estimate?.formatted?.expected || formatCurrency(results.estimated_cost_median, results.currency || formData.currency)}
                  </div>
                  <div className="text-sm text-gray-400 mt-1">expected cost</div>

                  <div className="mt-3 border-t border-gray-600/30 pt-3 space-y-2">
                    <div className="flex justify-between text-sm">
                      <div className="text-gray-400">Optimistic:</div>
                      <div className="font-semibold text-green-400">
                        {results.cost_estimate?.formatted?.optimistic || formatCurrency(results.cost_estimate?.optimistic, results.currency || formData.currency)}
                      </div>
                    </div>
                    <div className="flex justify-between text-sm">
                      <div className="text-gray-400">Conservative:</div>
                      <div className="font-semibold text-orange-400">
                        {results.cost_estimate?.formatted?.conservative || formatCurrency(results.estimated_cost_p90, results.currency || formData.currency)}
                      </div>
                    </div>
                  </div>
                </div>

                {results.interpretation && (
                  <div className="bg-gradient-to-br from-blue-900/30 to-blue-800/20 rounded-lg p-4 border border-blue-700/40 backdrop-blur-sm">
                    <div className="flex items-start gap-3">
                      <CheckCircle className="w-5 h-5 text-blue-400 flex-shrink-0 mt-0.5" />
                      <div>
                        <div className="font-semibold mb-2 text-blue-200">Clinical Interpretation</div>
                        {results.interpretation.message && (
                          <p className="text-sm text-gray-300 mb-2">{results.interpretation.message}</p>
                        )}
                        {results.interpretation.recommendation && (
                          <p className="text-sm text-gray-400 italic">
                            {results.interpretation.recommendation}
                          </p>
                        )}
                      </div>
                    </div>
                  </div>
                )}

                <div className="bg-gradient-to-br from-yellow-900/20 to-yellow-800/10 rounded-lg p-3 border border-yellow-700/30 backdrop-blur-sm">
                  <div className="flex gap-3">
                    <AlertCircle className="w-5 h-5 text-yellow-400 flex-shrink-0" />
                    <div className="text-sm text-yellow-200">
                      <div className="font-semibold mb-1">Clinical Judgment Required</div>
                      <div className="text-yellow-300/80">
                        Predictions are estimates based on historical data. Use alongside clinical assessment and judgment.
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="bg-gradient-to-br from-gray-900/90 to-gray-800/50 rounded-lg p-6 text-center border border-gray-700/50 backdrop-blur-sm">
                <div className="mx-auto mb-3 w-16 h-16 rounded-lg bg-gradient-to-br from-gray-700 to-gray-600 flex items-center justify-center shadow-lg">
                  <TrendingUp className="w-8 h-8 text-gray-300" />
                </div>
                <div className="text-lg font-semibold mb-1 text-gray-200">Ready to Predict</div>
                <div className="text-sm text-gray-400">
                  Enter patient information and generate a prediction to see AI-powered insights.
                </div>
              </div>
            )}
          </aside>
        </div>
      </main>
    </div>
  );
}