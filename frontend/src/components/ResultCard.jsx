import React, { useEffect, useRef } from 'react'
import { Link } from 'react-router-dom'

const riskColors = {
  high: { bg: '#fef2f2', border: '#fecaca', text: '#dc2626', light: '#fee2e2', accent: '#ef4444' },
  low: { bg: '#f0fdf4', border: '#bbf7d0', text: '#16a34a', light: '#dcfce7', accent: '#22c55e' },
}

const RiskGauge = ({ percentage, isHigh }) => {
  const color = isHigh ? '#ef4444' : '#22c55e'
  const radius = 70
  const stroke = 10
  const normalizedRadius = radius - stroke / 2
  const circumference = normalizedRadius * 2 * Math.PI
  const clamp = Math.min(Math.max(percentage, 0), 100)
  const dashoffset = circumference - (clamp / 100) * circumference

  return (
    <div style={{ position: 'relative', display: 'inline-flex', alignItems: 'center', justifyContent: 'center' }}>
      <svg width={radius * 2} height={radius * 2} viewBox={`0 0 ${radius * 2} ${radius * 2}`}>
        <circle
          cx={radius} cy={radius} r={normalizedRadius}
          fill="none" stroke="#e5e7eb" strokeWidth={stroke}
        />
        <circle
          cx={radius} cy={radius} r={normalizedRadius}
          fill="none" stroke={color} strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={`${circumference} ${circumference}`}
          strokeDashoffset={dashoffset}
          transform={`rotate(-90 ${radius} ${radius})`}
          style={{ transition: 'stroke-dashoffset 1.2s ease-out' }}
        />
      </svg>
      <div style={{
        position: 'absolute', textAlign: 'center',
      }}>
        <div style={{ fontSize: 22, fontWeight: 900, color, letterSpacing: '-1px' }}>
          {clamp.toFixed(1)}%
        </div>
        <div style={{ fontSize: 10, color: '#9ca3af', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px' }}>
          Risk Score
        </div>
      </div>
    </div>
  )
}

const DataRow = ({ label, value, highlight }) => (
  <div style={{
    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
    padding: '10px 0',
    borderBottom: '1px solid #f3f4f6',
  }}>
    <span style={{ fontSize: 13, color: '#6b7280' }}>{label}</span>
    <span style={{
      fontSize: 13, fontWeight: 600,
      color: highlight ? '#16a34a' : '#111827',
      background: highlight ? '#f0fdf4' : 'transparent',
      padding: highlight ? '2px 8px' : '0',
      borderRadius: 6,
    }}>{value}</span>
  </div>
)

export default function ResultCard({ result, onReset }) {
  const isHigh = result.prediction === 1
  const colors = isHigh ? riskColors.high : riskColors.low
  const pct = result.risk_percentage

  const confidenceBadge = {
    High:   { bg: '#dcfce7', text: '#15803d', border: '#bbf7d0' },
    Medium: { bg: '#fef9c3', text: '#854d0e', border: '#fde68a' },
    Low:    { bg: '#fee2e2', text: '#b91c1c', border: '#fecaca' },
  }[result.confidence] || {}

  const tips = isHigh
    ? ['Consult a cardiologist promptly', 'Monitor blood pressure daily', 'Adopt a heart-healthy diet', 'Increase physical activity gradually', 'Quit smoking immediately if applicable']
    : ['Maintain your healthy habits', 'Schedule annual checkups', 'Keep monitoring your cholesterol', 'Stay physically active', 'Limit alcohol and avoid smoking']

  return (
    <div style={{ animation: 'fadeIn 0.5s ease' }}>
      {/* Main result banner */}
      <div style={{
        background: `linear-gradient(135deg, ${colors.bg}, #fff)`,
        border: `2px solid ${colors.border}`,
        borderRadius: 20,
        padding: '36px',
        marginBottom: 20,
        boxShadow: `0 8px 32px ${isHigh ? 'rgba(239,68,68,0.08)' : 'rgba(34,197,94,0.08)'}`,
      }}>
        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 20, marginBottom: 32 }}>
          <div>
            <div style={{
              display: 'inline-flex', alignItems: 'center', gap: 6,
              fontSize: 11, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '1px',
              color: colors.text, marginBottom: 10,
            }}>
              {isHigh ? '🔴' : '🟢'} Assessment Complete
            </div>
            <h2 style={{
              fontSize: 30, fontWeight: 900,
              color: colors.text, letterSpacing: '-1px', marginBottom: 8,
            }}>
              {result.risk_level}
            </h2>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
              <span style={{
                fontSize: 12, fontWeight: 600, padding: '3px 10px',
                borderRadius: 99, background: confidenceBadge.bg,
                color: confidenceBadge.text, border: `1px solid ${confidenceBadge.border}`,
              }}>
                {result.confidence} Confidence
              </span>
              <span style={{ fontSize: 12, color: '#9ca3af' }}>
                Logistic Regression Model v1.0
              </span>
            </div>
          </div>
          <RiskGauge percentage={pct} isHigh={isHigh} />
        </div>

        {/* Message */}
        <div style={{
          background: '#fff',
          border: `1px solid ${colors.border}`,
          borderRadius: 12, padding: '16px 20px',
          fontSize: 14, color: '#374151', lineHeight: 1.7,
          marginBottom: 24,
        }}>
          {result.message}
        </div>

        {/* Risk bar */}
        <div style={{ marginBottom: 8 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
            <span style={{ fontSize: 12, fontWeight: 600, color: '#6b7280' }}>Risk Level</span>
            <span style={{ fontSize: 12, fontWeight: 700, color: colors.text }}>{pct.toFixed(2)}%</span>
          </div>
          <div style={{ height: 10, background: '#e5e7eb', borderRadius: 99, overflow: 'hidden' }}>
            <div style={{
              height: '100%',
              width: `${pct}%`,
              background: isHigh
                ? 'linear-gradient(90deg, #fca5a5, #ef4444)'
                : 'linear-gradient(90deg, #86efac, #22c55e)',
              borderRadius: 99,
              transition: 'width 1.2s ease-out',
            }}></div>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4 }}>
            <span style={{ fontSize: 11, color: '#9ca3af' }}>0% (No Risk)</span>
            <span style={{ fontSize: 11, color: '#9ca3af' }}>100% (Extreme)</span>
          </div>
        </div>
      </div>

      {/* Input Summary + Tips */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 20 }}>
        {/* Input summary */}
        <div style={{
          background: '#fff', border: '1px solid #e5e7eb',
          borderRadius: 16, padding: '24px',
          boxShadow: '0 2px 8px rgba(0,0,0,0.04)',
        }}>
          <h3 style={{ fontSize: 14, fontWeight: 700, color: '#374151', marginBottom: 16, textTransform: 'uppercase', letterSpacing: '0.5px' }}>
            📋 Your Input Data
          </h3>
          <DataRow label="Age" value={`${result.input_summary.age} years`} />
          <DataRow label="Gender" value={result.input_summary.gender} />
          <DataRow label="Cholesterol" value={`${result.input_summary.cholesterol} mg/dL`} highlight={result.input_summary.cholesterol > 200} />
          <DataRow label="Blood Pressure" value={`${result.input_summary.blood_pressure} mmHg`} highlight={result.input_summary.blood_pressure > 130} />
          <DataRow label="Glucose" value={`${result.input_summary.glucose} mg/dL`} highlight={result.input_summary.glucose > 100} />
          <DataRow label="BMI" value={result.input_summary.bmi.toFixed(1)} highlight={result.input_summary.bmi > 25} />
          <DataRow label="Smoking" value={result.input_summary.smoking ? '🚬 Yes' : '🚭 No'} />
          <DataRow label="Alcohol" value={result.input_summary.alcohol ? '🍺 Yes' : '🚫 No'} />
          <DataRow label="Physically Active" value={result.input_summary.physical_activity ? '🏃 Yes' : '🛋 No'} />
        </div>

        {/* Recommendations */}
        <div style={{
          background: '#fff', border: '1px solid #e5e7eb',
          borderRadius: 16, padding: '24px',
          boxShadow: '0 2px 8px rgba(0,0,0,0.04)',
        }}>
          <h3 style={{ fontSize: 14, fontWeight: 700, color: '#374151', marginBottom: 16, textTransform: 'uppercase', letterSpacing: '0.5px' }}>
            {isHigh ? '⚕️ Recommendations' : '✅ Keep It Up'}
          </h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {tips.map((tip, i) => (
              <div key={i} style={{
                display: 'flex', gap: 10, alignItems: 'flex-start',
                padding: '10px 12px',
                background: isHigh ? '#fef2f2' : '#f0fdf4',
                borderRadius: 10,
                border: `1px solid ${isHigh ? '#fecaca' : '#bbf7d0'}`,
              }}>
                <span style={{ fontSize: 14, flexShrink: 0, marginTop: 1 }}>
                  {isHigh ? '⚠️' : '✓'}
                </span>
                <span style={{ fontSize: 13, color: isHigh ? '#7f1d1d' : '#14532d', lineHeight: 1.5 }}>
                  {tip}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Action buttons */}
      <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap' }}>
        <button
          onClick={onReset}
          style={{
            flex: 1, padding: '14px',
            borderRadius: 12,
            background: 'linear-gradient(135deg, #22c55e, #16a34a)',
            color: '#fff', fontWeight: 700, fontSize: 15,
            boxShadow: '0 4px 14px rgba(34,197,94,0.3)',
            cursor: 'pointer', border: 'none',
            transition: 'all 0.25s',
          }}
          onMouseEnter={e => e.currentTarget.style.transform = 'translateY(-2px)'}
          onMouseLeave={e => e.currentTarget.style.transform = 'translateY(0)'}
        >
          ↩ New Assessment
        </button>
        <Link to="/" style={{
          flex: 1, padding: '14px',
          borderRadius: 12,
          background: '#fff',
          color: '#374151', fontWeight: 600, fontSize: 15,
          border: '2px solid #e5e7eb',
          textAlign: 'center',
          transition: 'all 0.25s',
          display: 'block',
        }}
        onMouseEnter={e => { e.currentTarget.style.borderColor = '#86efac'; e.currentTarget.style.color = '#16a34a' }}
        onMouseLeave={e => { e.currentTarget.style.borderColor = '#e5e7eb'; e.currentTarget.style.color = '#374151' }}
        >
          ← Back to Home
        </Link>
      </div>

      <div style={{ textAlign: 'center', marginTop: 20, fontSize: 12, color: '#9ca3af' }}>
        ⚕️ This result is generated by a machine learning model and is not a medical diagnosis.
      </div>
    </div>
  )
}
