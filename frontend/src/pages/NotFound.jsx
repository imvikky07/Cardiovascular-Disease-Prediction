import React from 'react'
import { Link } from 'react-router-dom'

export default function NotFound() {
  return (
    <div style={{
      paddingTop: 68, minHeight: '100vh',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      background: '#f9fafb',
    }}>
      <div style={{ textAlign: 'center', padding: '40px 24px' }}>
        <div style={{ fontSize: 80, marginBottom: 20 }}>🫀</div>
        <h1 style={{ fontSize: 80, fontWeight: 900, color: '#e5e7eb', letterSpacing: '-4px', marginBottom: 0, lineHeight: 1 }}>
          404
        </h1>
        <h2 style={{ fontSize: 24, fontWeight: 700, color: '#111827', margin: '16px 0 12px', letterSpacing: '-0.5px' }}>
          Page Not Found
        </h2>
        <p style={{ color: '#6b7280', marginBottom: 32, fontSize: 15 }}>
          The page you're looking for doesn't exist or has been moved.
        </p>
        <div style={{ display: 'flex', gap: 14, justifyContent: 'center', flexWrap: 'wrap' }}>
          <Link to="/" style={{
            padding: '12px 28px', borderRadius: 10,
            background: 'linear-gradient(135deg, #22c55e, #16a34a)',
            color: '#fff', fontWeight: 600, fontSize: 15,
            boxShadow: '0 4px 12px rgba(34,197,94,0.3)',
          }}>
            Go Home
          </Link>
          <Link to="/predict" style={{
            padding: '12px 28px', borderRadius: 10,
            background: '#fff', color: '#374151',
            fontWeight: 600, fontSize: 15,
            border: '2px solid #e5e7eb',
          }}>
            Risk Assessment
          </Link>
        </div>
      </div>
    </div>
  )
}
