import React, { useState } from 'react';

export default function FAQSection() {
  const [openIndex, setOpenIndex] = useState(null);

  const faqs = [
    {
      question: "How accurate are your predictions for hospital LOS?",
      answer: "Our dual-model system achieves high accuracy through specialized routing. The Generalist model handles low-to-medium risk patients (SAPS-II ≤29) while the Specialist model focuses on high-risk cases (SAPS-II >29). We provide three prediction levels: P10 (optimistic), P50 (median/expected), and P90 (conservative), giving you an 80% confidence interval. The system is trained on 431,241+ ICU patient records from MIMIC-IV database, incorporating clinical metrics like SAPS-II scores, lab values (creatinine, hemoglobin), vital signs, and diagnosis categories. Our P90 predictions typically show 85-95% coverage, meaning actual LOS falls below the P90 estimate in 9 out of 10 cases."
    },
    {
      question: "What makes the AI 'explainable'?",
      answer: "Our system provides complete transparency through: Feature Importance - You can see exactly which patient characteristics influenced each prediction (SAPS-II score, diagnosis category, lab values, etc.). Model Routing Visibility - The system explicitly shows whether the Generalist or Specialist model was used and why. Confidence Intervals - P10-P90 ranges communicate prediction uncertainty. Clinical Interpretation - Context-specific recommendations explain the prediction in clinical terms. Diagnosis Categorization - Automatic mapping of diagnoses to clinical categories is shown and can be verified. This explainability ensures clinicians understand not just what the AI predicts, but why it made that prediction."
    },
    {
      question: "How is this solution deployed in a real hospital?",
      answer: "Deployment follows a flexible architecture: API Integration - RESTful API endpoints (/predict for single patients, /predict_batch for bulk processing, /forecast_occupancy for capacity planning) integrate with existing Hospital Information Systems (HIS) or Electronic Health Records (EHR). Real-time Web Interface - Healthcare providers access predictions through our responsive web application, entering patient data manually or via automated imports. Batch Processing - Process entire patient cohorts (up to 100 patients per request) for admission planning and capacity forecasting. The system supports single sign-on (SSO), HIPAA-compliant data handling, and secure cloud deployment (AWS, Azure, GCP) or on-premises installation. Models are served via Flask API (production typically uses Gunicorn/uWSGI) with load balancing for high availability."
    },
    {
      question: "What features does the model use for predictions?",
      answer: "The model analyzes comprehensive patient data including: Demographics (age, gender), Admission details (type: emergency/urgent/elective, insurance), Clinical severity (SAPS-II score - a validated ICU mortality predictor), Procedures performed during stay, Laboratory values (maximum creatinine for kidney function, minimum hemoglobin for anemia), Vital signs (average heart rate), and Primary diagnosis with automatic categorization into 9 clinical categories (infection, cardiovascular, respiratory, gastrointestinal, neurological, renal, trauma, cancer, metabolic). The model also uses engineered features like age-heart rate interactions to capture complex clinical patterns."
    },
    {
      question: "What do the P10, P50, and P90 predictions mean?",
      answer: "These percentile predictions provide a complete picture of expected outcomes: P10 (Optimistic) - Only 10% of patients stay less than this duration; represents a best-case scenario. P50 (Median/Expected) - Half of similar patients stay shorter, half stay longer; this is our primary prediction and most likely outcome. P90 (Conservative) - 90% of patients are discharged before this point; useful for resource planning and worst-case scenarios. The range between P10 and P90 represents your 80% confidence interval. For example, if predictions are 2 days (P10), 5 days (P50), and 11 days (P90), you can expect the stay to most likely be around 5 days, with an 80% chance of being between 2-11 days."
    },
    {
      question: "How are cost estimates calculated?",
      answer: "Cost projections multiply predicted LOS by your hospital's average daily patient cost. You can customize the daily cost and currency (INR, USD, EUR, GBP) to match your facility. We provide three cost scenarios: Optimistic cost (P10 LOS × daily cost) for best-case budgeting, Expected cost (P50 LOS × daily cost) for typical resource allocation, and Conservative cost (P90 LOS × daily cost) for worst-case financial planning. The system formats currency appropriately (e.g., ₹2,50,000 for INR) and provides both raw numbers and formatted displays. This helps finance teams with insurance pre-authorization, budget forecasting, and resource allocation decisions."
    },
    {
      question: "What clinical safeguards are built into the system?",
      answer: "Multiple safety layers ensure responsible clinical use: Explicit disclaimers remind clinicians that predictions are estimates requiring clinical judgment verification. Risk level indicators (Low/Medium/High) based on SAPS-II scores alert providers to patient acuity. Clinical interpretation messages provide context-specific recommendations (e.g., 'High-risk patient with extended stay expected. Consider early intervention'). Confidence intervals (P10-P90 range) communicate prediction uncertainty transparently. The system never claims diagnostic capability - it supports, not replaces, clinical decision-making. All predictions display which model (Generalist/Specialist) was used for auditability. We recommend using predictions alongside: physician clinical assessment, real-time patient condition monitoring, and interdisciplinary team discussions."
    },
    {
      question: "What data privacy and security measures are in place?",
      answer: "Our system implements healthcare-grade security: Data Encryption - All API communications use HTTPS/TLS encryption; patient data is encrypted at rest and in transit. No Data Storage - The API is stateless; no patient identifiable information (PII) is stored after prediction generation. Anonymization - Patient identifiers (names, MRNs) are never required; the model works with clinical features only. Audit Logging - All API requests can be logged (optional) with timestamps for compliance auditing, without storing patient details. CORS Protection - Cross-origin resource sharing configured to allow only authorized domains. The deployment supports: HIPAA compliance when hosted appropriately, SOC 2 compliant infrastructure options, and integration with hospital security systems (Active Directory, LDAP). For maximum privacy, hospitals can deploy the solution entirely within their internal network with no external data transmission."
    }
  ];

  const toggleFAQ = (index) => {
    setOpenIndex(openIndex === index ? null : index);
  };

  return (
    <section className="py-32 bg-black">
      <div className="max-w-2xl mx-auto px-6">
        <h3 className="text-2xl font-bold mb-12 text-center text-white">
          Frequently asked questions
        </h3>

        <div className="space-y-6">
          {faqs.map((faq, index) => (
            <div key={index}>
              <button
                onClick={() => toggleFAQ(index)}
                className="w-full flex items-center justify-between gap-4 px-6 py-4 bg-neutral-900/40 rounded-full text-left hover:bg-neutral-800/60 transition-all"
              >
                <span className="text-white text-left">{faq.question}</span>
                <span className="text-xl text-white flex-shrink-0 transition-transform duration-200" style={{
                  transform: openIndex === index ? 'rotate(45deg)' : 'rotate(0deg)'
                }}>
                  +
                </span>
              </button>
              
              <div
                className="overflow-hidden transition-all duration-300 ease-in-out"
                style={{
                  maxHeight: openIndex === index ? '1000px' : '0px',
                  opacity: openIndex === index ? 1 : 0
                }}
              >
                <div className="px-6 pt-4 pb-2">
                  <p className="text-gray-400 leading-relaxed text-sm">
                    {faq.answer}
                  </p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}