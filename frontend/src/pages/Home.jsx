import React, { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { checkHealth } from '../utils/api'

const StatCard = ({ value, label, icon }) => (
  <div style={{
    background: '#fff',
    border: '1px solid #e5e7eb',
    borderRadius: 16,
    padding: '28px 32px',
    textAlign: 'center',
    boxShadow: '0 2px 8px rgba(0,0,0,0.04)',
    transition: 'all 0.3s',
  }}
  onMouseEnter={e => { e.currentTarget.style.transform = 'translateY(-4px)'; e.currentTarget.style.boxShadow = '0 12px 28px rgba(34,197,94,0.12)'; e.currentTarget.style.borderColor = '#bbf7d0' }}
  onMouseLeave={e => { e.currentTarget.style.transform = 'translateY(0)'; e.currentTarget.style.boxShadow = '0 2px 8px rgba(0,0,0,0.04)'; e.currentTarget.style.borderColor = '#e5e7eb' }}
  >
    <div style={{ fontSize: 32, marginBottom: 8 }}>{icon}</div>
    <div style={{ fontSize: 32, fontWeight: 800, color: '#22c55e', letterSpacing: '-1px' }}>{value}</div>
    <div style={{ fontSize: 13, color: '#6b7280', marginTop: 4 }}>{label}</div>
  </div>
)

const FeatureCard = ({ icon, title, desc }) => (
  <div style={{
    background: '#fff',
    border: '1px solid #e5e7eb',
    borderRadius: 16,
    padding: '28px',
    boxShadow: '0 2px 8px rgba(0,0,0,0.04)',
    transition: 'all 0.3s',
  }}
  onMouseEnter={e => { e.currentTarget.style.borderColor = '#86efac'; e.currentTarget.style.boxShadow = '0 8px 24px rgba(34,197,94,0.1)' }}
  onMouseLeave={e => { e.currentTarget.style.borderColor = '#e5e7eb'; e.currentTarget.style.boxShadow = '0 2px 8px rgba(0,0,0,0.04)' }}
  >
    <div style={{
      width: 48, height: 48, borderRadius: 12,
      background: 'linear-gradient(135deg, #f0fdf4, #dcfce7)',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      fontSize: 22, marginBottom: 16,
    }}>{icon}</div>
    <div style={{ fontWeight: 700, fontSize: 16, color: '#111827', marginBottom: 8 }}>{title}</div>
    <div style={{ fontSize: 14, color: '#6b7280', lineHeight: 1.6 }}>{desc}</div>
  </div>
)

const StepBadge = ({ n }) => (
  <div style={{
    width: 36, height: 36, borderRadius: '50%',
    background: 'linear-gradient(135deg, #22c55e, #16a34a)',
    color: '#fff', fontWeight: 700, fontSize: 15,
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    flexShrink: 0, boxShadow: '0 2px 8px rgba(34,197,94,0.4)',
  }}>{n}</div>
)

export default function Home() {
  const [apiStatus, setApiStatus] = useState(null)

  useEffect(() => {
    checkHealth().then(setApiStatus).catch(() => setApiStatus(null))
  }, [])

  const stats = [
    { value: '1,500+', label: 'Training Records', icon: '📊' },
    { value: '78.7%', label: 'Model Accuracy', icon: '🎯' },
    { value: '9', label: 'Risk Factors', icon: '🔬' },
    { value: '<1s', label: 'Prediction Time', icon: '⚡' },
  ]

  const features = [
    { icon: '🤖', title: 'Logistic Regression', desc: 'Interpretable, production-grade ML model trained on clinical cardiovascular data with StandardScaler preprocessing.' },
    { icon: '🩺', title: 'Clinical Risk Factors', desc: 'Evaluates age, gender, cholesterol, blood pressure, glucose, BMI, smoking, alcohol, and physical activity.' },
    { icon: '📈', title: 'Probability Score', desc: 'Returns a precise CVD risk percentage, binary classification, and confidence level for each prediction.' },
    { icon: '🔒', title: 'Privacy First', desc: 'All predictions run server-side. No personal data stored. Inputs are never logged or retained.' },
    { icon: '⚡', title: 'Instant Results', desc: 'FastAPI backend delivers sub-second predictions with full validation and structured error handling.' },
    { icon: '📱', title: 'Fully Responsive', desc: 'Works seamlessly on desktop, tablet, and mobile for accessible heart health screening anywhere.' },
  ]

  return (
    <div style={{ paddingTop: 68 }}>
      {/* Hero */}
      <section style={{
        background: 'linear-gradient(160deg, #f0fdf4 0%, #fff 50%, #f0fdf4 100%)',
        padding: '80px 24px 80px',
        textAlign: 'center',
        borderBottom: '1px solid #e5e7eb',
      }}>
        {apiStatus && (
          <div style={{
            display: 'inline-flex', alignItems: 'center', gap: 6,
            background: '#dcfce7', border: '1px solid #bbf7d0',
            borderRadius: 99, padding: '5px 14px',
            fontSize: 12, color: '#15803d', fontWeight: 600,
            marginBottom: 28,
          }}>
            <span style={{ width: 7, height: 7, borderRadius: '50%', background: '#22c55e', display: 'inline-block', boxShadow: '0 0 0 2px rgba(34,197,94,0.3)' }}></span>
            API Online · Model Ready
          </div>
        )}

        <h1 style={{
          fontSize: 'clamp(36px, 6vw, 64px)',
          fontWeight: 900,
          color: '#111827',
          lineHeight: 1.1,
          letterSpacing: '-2px',
          maxWidth: 700,
          margin: '0 auto 24px',
        }}>
          Predict Your<br />
          <span style={{ color: '#22c55e' }}>Cardiovascular Risk</span><br />
          with AI
        </h1>

        <p style={{
          fontSize: 18, color: '#6b7280',
          maxWidth: 520, margin: '0 auto 40px',
          lineHeight: 1.7,
        }}>
          Enter 9 clinical markers and receive an instant, AI-powered cardiovascular disease risk assessment powered by logistic regression.
        </p>

        <div style={{ display: 'flex', gap: 14, justifyContent: 'center', flexWrap: 'wrap' }}>
          <Link to="/predict" style={{
            padding: '14px 32px',
            borderRadius: 12,
            background: 'linear-gradient(135deg, #22c55e, #16a34a)',
            color: '#fff',
            fontWeight: 700, fontSize: 16,
            boxShadow: '0 4px 16px rgba(34,197,94,0.35)',
            transition: 'all 0.25s',
            display: 'inline-block',
          }}
          onMouseEnter={e => { e.currentTarget.style.transform = 'translateY(-2px)'; e.currentTarget.style.boxShadow = '0 8px 24px rgba(34,197,94,0.45)' }}
          onMouseLeave={e => { e.currentTarget.style.transform = 'translateY(0)'; e.currentTarget.style.boxShadow = '0 4px 16px rgba(34,197,94,0.35)' }}
          >
            Start Risk Assessment →
          </Link>
          <Link to="/about" style={{
            padding: '14px 32px',
            borderRadius: 12,
            background: '#fff',
            color: '#374151',
            fontWeight: 600, fontSize: 16,
            border: '2px solid #e5e7eb',
            transition: 'all 0.25s',
            display: 'inline-block',
          }}
          onMouseEnter={e => { e.currentTarget.style.borderColor = '#86efac'; e.currentTarget.style.color = '#16a34a' }}
          onMouseLeave={e => { e.currentTarget.style.borderColor = '#e5e7eb'; e.currentTarget.style.color = '#374151' }}
          >
            Learn More
          </Link>
        </div>

        {/* Disclaimer */}
        <p style={{ fontSize: 12, color: '#9ca3af', marginTop: 28 }}>
          ⚕️ For educational purposes only. Not a substitute for professional medical advice.
        </p>
      </section>

      {/* Stats */}
      <section style={{ padding: '60px 24px', maxWidth: 1000, margin: '0 auto' }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 20 }}>
          {stats.map(s => <StatCard key={s.label} {...s} />)}
        </div>
      </section>

      {/* How it works */}
      <section style={{ padding: '20px 24px 60px', maxWidth: 800, margin: '0 auto' }}>
        <h2 style={{ fontSize: 28, fontWeight: 800, color: '#111827', textAlign: 'center', marginBottom: 8, letterSpacing: '-0.5px' }}>
          How It Works
        </h2>
        <p style={{ textAlign: 'center', color: '#6b7280', marginBottom: 40, fontSize: 15 }}>
          Three steps to your cardiovascular risk report
        </p>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          {[
            ['Enter Clinical Data', 'Provide 9 health markers including age, cholesterol, blood pressure, glucose, BMI, and lifestyle factors.'],
            ['AI Analysis', 'Our logistic regression model, trained on 1,500 records, analyzes your inputs through a validated clinical pipeline.'],
            ['Instant Risk Report', 'Receive your CVD risk percentage, Low/High classification, confidence level, and personalized advice.'],
          ].map(([title, desc], i) => (
            <div key={i} style={{
              display: 'flex', gap: 20, alignItems: 'flex-start',
              background: '#fff', border: '1px solid #e5e7eb',
              borderRadius: 14, padding: '24px',
              boxShadow: '0 2px 8px rgba(0,0,0,0.03)',
            }}>
              <StepBadge n={i + 1} />
              <div>
                <div style={{ fontWeight: 700, fontSize: 16, color: '#111827', marginBottom: 6 }}>{title}</div>
                <div style={{ fontSize: 14, color: '#6b7280', lineHeight: 1.6 }}>{desc}</div>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Features */}
      <section style={{
        padding: '60px 24px',
        background: '#f9fafb',
        borderTop: '1px solid #e5e7eb',
        borderBottom: '1px solid #e5e7eb',
      }}>
        <div style={{ maxWidth: 1100, margin: '0 auto' }}>
          <h2 style={{ fontSize: 28, fontWeight: 800, color: '#111827', textAlign: 'center', marginBottom: 8, letterSpacing: '-0.5px' }}>
            Built for Accuracy & Trust
          </h2>
          <p style={{ textAlign: 'center', color: '#6b7280', marginBottom: 48, fontSize: 15 }}>
            Every component designed for production-grade reliability
          </p>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 20 }}>
            {features.map(f => <FeatureCard key={f.title} {...f} />)}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section style={{ padding: '80px 24px', textAlign: 'center' }}>
        <div style={{
          maxWidth: 560, margin: '0 auto',
          background: 'linear-gradient(135deg, #f0fdf4, #dcfce7)',
          border: '1px solid #bbf7d0',
          borderRadius: 24, padding: '48px 40px',
        }}>
          <div style={{ fontSize: 40, marginBottom: 16 }}>❤️</div>
          <h2 style={{ fontSize: 28, fontWeight: 800, color: '#111827', marginBottom: 12, letterSpacing: '-0.5px' }}>
            Ready to check your heart health?
          </h2>
          <p style={{ color: '#6b7280', marginBottom: 28, lineHeight: 1.6 }}>
            Takes less than 2 minutes. Get your personalized CVD risk score now.
          </p>
          <Link to="/predict" style={{
            display: 'inline-block',
            padding: '14px 36px',
            borderRadius: 12,
            background: 'linear-gradient(135deg, #22c55e, #16a34a)',
            color: '#fff',
            fontWeight: 700, fontSize: 16,
            boxShadow: '0 4px 16px rgba(34,197,94,0.3)',
            transition: 'all 0.25s',
          }}
          onMouseEnter={e => { e.currentTarget.style.transform = 'translateY(-2px)' }}
          onMouseLeave={e => { e.currentTarget.style.transform = 'translateY(0)' }}
          >
            Get My Risk Score →
          </Link>
        </div>
      </section>
    </div>
  )
}
