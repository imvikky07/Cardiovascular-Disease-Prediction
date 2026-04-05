import React from 'react'
import { Link } from 'react-router-dom'

export default function Footer() {
  return (
    <footer style={{
      background: '#f9fafb',
      borderTop: '1px solid #e5e7eb',
      padding: '40px 24px 24px',
      marginTop: 'auto',
    }}>
      <div style={{ maxWidth: 1200, margin: '0 auto' }}>
        <div style={{
          display: 'flex', flexWrap: 'wrap',
          justifyContent: 'space-between', gap: 32,
          marginBottom: 32,
        }}>
          <div style={{ maxWidth: 280 }}>
            <div style={{ fontWeight: 800, fontSize: 18, color: '#111827', marginBottom: 8 }}>
              CVD<span style={{ color: '#22c55e' }}>Risk</span>
            </div>
            <p style={{ fontSize: 13, color: '#6b7280', lineHeight: 1.6 }}>
              AI-powered cardiovascular disease risk assessment using logistic regression trained on clinical data.
            </p>
          </div>
          <div style={{ display: 'flex', gap: 48, flexWrap: 'wrap' }}>
            <div>
              <div style={{ fontWeight: 600, fontSize: 13, color: '#374151', marginBottom: 12, textTransform: 'uppercase', letterSpacing: '0.5px' }}>App</div>
              {[['/', 'Home'], ['/predict', 'Risk Assessment'], ['/about', 'About']].map(([to, label]) => (
                <div key={to} style={{ marginBottom: 8 }}>
                  <Link to={to} style={{ fontSize: 13, color: '#6b7280', transition: 'color 0.2s' }}
                    onMouseEnter={e => e.target.style.color = '#22c55e'}
                    onMouseLeave={e => e.target.style.color = '#6b7280'}
                  >{label}</Link>
                </div>
              ))}
            </div>
            <div>
              <div style={{ fontWeight: 600, fontSize: 13, color: '#374151', marginBottom: 12, textTransform: 'uppercase', letterSpacing: '0.5px' }}>Tech Stack</div>
              {['FastAPI', 'React + Vite', 'Scikit-learn', 'Docker'].map(t => (
                <div key={t} style={{ fontSize: 13, color: '#6b7280', marginBottom: 8 }}>{t}</div>
              ))}
            </div>
          </div>
        </div>
        <div style={{
          borderTop: '1px solid #e5e7eb', paddingTop: 20,
          display: 'flex', flexWrap: 'wrap',
          justifyContent: 'space-between', alignItems: 'center', gap: 12,
        }}>
          <span style={{ fontSize: 12, color: '#9ca3af' }}>
            © 2025 CVDRisk. For educational purposes only — not a substitute for medical advice.
          </span>
          <span style={{ fontSize: 12, color: '#9ca3af' }}>
            Built with React + FastAPI + Logistic Regression
          </span>
        </div>
      </div>
    </footer>
  )
}
