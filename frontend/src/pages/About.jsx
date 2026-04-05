import React from 'react'
import { Link } from 'react-router-dom'

const InfoCard = ({ icon, title, children }) => (
  <div style={{
    background: '#fff', border: '1px solid #e5e7eb',
    borderRadius: 16, padding: '28px',
    boxShadow: '0 2px 8px rgba(0,0,0,0.04)',
    marginBottom: 20,
  }}>
    <div style={{ display: 'flex', gap: 14, alignItems: 'flex-start' }}>
      <div style={{
        fontSize: 22, width: 46, height: 46,
        background: 'linear-gradient(135deg, #f0fdf4, #dcfce7)',
        borderRadius: 12, display: 'flex', alignItems: 'center', justifyContent: 'center',
        flexShrink: 0,
      }}>{icon}</div>
      <div>
        <h3 style={{ fontSize: 17, fontWeight: 700, color: '#111827', marginBottom: 8 }}>{title}</h3>
        {children}
      </div>
    </div>
  </div>
)

const FactorRow = ({ factor, normal, risky }) => (
  <tr>
    <td style={{ padding: '10px 14px', fontSize: 13, fontWeight: 600, color: '#374151', borderBottom: '1px solid #f3f4f6' }}>{factor}</td>
    <td style={{ padding: '10px 14px', fontSize: 13, color: '#16a34a', borderBottom: '1px solid #f3f4f6' }}>{normal}</td>
    <td style={{ padding: '10px 14px', fontSize: 13, color: '#dc2626', borderBottom: '1px solid #f3f4f6' }}>{risky}</td>
  </tr>
)

export default function About() {
  return (
    <div style={{ paddingTop: 68, minHeight: '100vh', background: '#f9fafb' }}>
      <div style={{ maxWidth: 820, margin: '0 auto', padding: '48px 24px 80px' }}>
        {/* Hero */}
        <div style={{ textAlign: 'center', marginBottom: 48 }}>
          <div style={{
            display: 'inline-flex', alignItems: 'center', gap: 6,
            background: '#dcfce7', border: '1px solid #bbf7d0',
            borderRadius: 99, padding: '5px 14px',
            fontSize: 12, color: '#15803d', fontWeight: 600, marginBottom: 16,
          }}>
            🔬 About This Project
          </div>
          <h1 style={{ fontSize: 36, fontWeight: 900, color: '#111827', letterSpacing: '-1px', marginBottom: 14 }}>
            CVD Risk Detection
          </h1>
          <p style={{ fontSize: 16, color: '#6b7280', lineHeight: 1.7, maxWidth: 560, margin: '0 auto' }}>
            A production-grade machine learning application for cardiovascular disease risk prediction using logistic regression trained on clinical data.
          </p>
        </div>

        {/* Model Info */}
        <InfoCard icon="🤖" title="Machine Learning Model">
          <p style={{ fontSize: 14, color: '#6b7280', lineHeight: 1.7, marginBottom: 12 }}>
            This application uses a <strong>Logistic Regression</strong> classifier — a well-established, interpretable algorithm widely used in clinical decision support. The model was trained on 1,500 synthetic records derived from established CVD risk factor relationships validated in clinical literature.
          </p>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
            {[
              ['Algorithm', 'Logistic Regression'],
              ['Training Records', '1,500'],
              ['Test Accuracy', '78.7%'],
              ['Preprocessing', 'StandardScaler + LabelEncoder'],
              ['Class Balancing', 'class_weight="balanced"'],
              ['Framework', 'scikit-learn 1.5'],
            ].map(([k, v]) => (
              <div key={k} style={{ background: '#f9fafb', borderRadius: 8, padding: '10px 14px', border: '1px solid #e5e7eb' }}>
                <div style={{ fontSize: 11, color: '#9ca3af', marginBottom: 2, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.4px' }}>{k}</div>
                <div style={{ fontSize: 13, fontWeight: 700, color: '#22c55e' }}>{v}</div>
              </div>
            ))}
          </div>
        </InfoCard>

        {/* Risk Factors */}
        <InfoCard icon="🩺" title="Clinical Risk Factors Evaluated">
          <p style={{ fontSize: 14, color: '#6b7280', lineHeight: 1.7, marginBottom: 14 }}>
            The model evaluates 9 validated cardiovascular risk factors:
          </p>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: '#f9fafb' }}>
                  <th style={{ padding: '10px 14px', textAlign: 'left', color: '#374151', fontWeight: 700, fontSize: 12 }}>Factor</th>
                  <th style={{ padding: '10px 14px', textAlign: 'left', color: '#16a34a', fontWeight: 700, fontSize: 12 }}>✅ Normal</th>
                  <th style={{ padding: '10px 14px', textAlign: 'left', color: '#dc2626', fontWeight: 700, fontSize: 12 }}>⚠️ Risky</th>
                </tr>
              </thead>
              <tbody>
                <FactorRow factor="Age" normal="< 45 years" risky="> 60 years" />
                <FactorRow factor="Cholesterol" normal="< 200 mg/dL" risky="> 240 mg/dL" />
                <FactorRow factor="Blood Pressure" normal="< 120 mmHg" risky="> 140 mmHg" />
                <FactorRow factor="Glucose" normal="< 100 mg/dL" risky="> 126 mg/dL" />
                <FactorRow factor="BMI" normal="18.5 – 24.9" risky="> 30 (obese)" />
                <FactorRow factor="Smoking" normal="Non-smoker" risky="Active smoker" />
                <FactorRow factor="Alcohol" normal="Non-drinker" risky="Regular intake" />
                <FactorRow factor="Physical Activity" normal="Active" risky="Sedentary" />
              </tbody>
            </table>
          </div>
        </InfoCard>

        {/* Tech Stack */}
        <InfoCard icon="⚙️" title="Technology Stack">
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
            {[
              { label: 'Backend', items: ['FastAPI (Python)', 'Uvicorn ASGI server', 'Pydantic v2 validation', 'scikit-learn'] },
              { label: 'Frontend', items: ['React 18 + Vite', 'React Router v6', 'React Hook Form', 'Axios'] },
              { label: 'ML Pipeline', items: ['Logistic Regression', 'StandardScaler', 'LabelEncoder', 'joblib persistence'] },
              { label: 'Deployment', items: ['Docker + Docker Compose', 'Environment variables', 'CORS configured', 'Health endpoint'] },
            ].map(({ label, items }) => (
              <div key={label} style={{ background: '#f9fafb', borderRadius: 10, padding: '14px 16px', border: '1px solid #e5e7eb' }}>
                <div style={{ fontSize: 12, fontWeight: 700, color: '#22c55e', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 8 }}>{label}</div>
                {items.map(item => (
                  <div key={item} style={{ fontSize: 13, color: '#374151', marginBottom: 4, display: 'flex', gap: 6, alignItems: 'center' }}>
                    <span style={{ color: '#22c55e', fontWeight: 700 }}>·</span> {item}
                  </div>
                ))}
              </div>
            ))}
          </div>
        </InfoCard>

        {/* Disclaimer */}
        <div style={{
          background: '#fffbeb', border: '1px solid #fde68a',
          borderRadius: 14, padding: '20px 24px',
          marginBottom: 28,
        }}>
          <div style={{ fontWeight: 700, color: '#92400e', fontSize: 14, marginBottom: 6 }}>⚠️ Medical Disclaimer</div>
          <p style={{ fontSize: 13, color: '#92400e', lineHeight: 1.7 }}>
            This tool is for <strong>educational and demonstration purposes only</strong>. It is not intended to diagnose, treat, cure, or prevent any medical condition. Predictions are generated by a statistical model and should never replace professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider with any questions about your cardiovascular health.
          </p>
        </div>

        {/* CTA */}
        <div style={{ textAlign: 'center' }}>
          <Link to="/predict" style={{
            display: 'inline-block',
            padding: '14px 36px',
            borderRadius: 12,
            background: 'linear-gradient(135deg, #22c55e, #16a34a)',
            color: '#fff', fontWeight: 700, fontSize: 15,
            boxShadow: '0 4px 14px rgba(34,197,94,0.3)',
            transition: 'all 0.25s',
          }}
          onMouseEnter={e => e.currentTarget.style.transform = 'translateY(-2px)'}
          onMouseLeave={e => e.currentTarget.style.transform = 'translateY(0)'}
          >
            Try the Risk Assessment →
          </Link>
        </div>
      </div>
    </div>
  )
}
