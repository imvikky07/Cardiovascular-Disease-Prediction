import React, { useState, useEffect } from 'react'
import { Link, useLocation } from 'react-router-dom'

const HeartIcon = () => (
  <svg width="28" height="28" viewBox="0 0 24 24" fill="none">
    <path d="M12 21C12 21 3 14 3 8.5C3 5.46 5.46 3 8.5 3C10.24 3 11.91 3.81 13 5.08C14.09 3.81 15.76 3 17.5 3C20.54 3 23 5.46 23 8.5C23 14 12 21 12 21Z" fill="#22c55e"/>
    <polyline points="4,13 8,9 11,14 15,6 18,11 22,8" fill="none" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
)

export default function Navbar() {
  const { pathname } = useLocation()
  const [scrolled, setScrolled] = useState(false)

  useEffect(() => {
    const handler = () => setScrolled(window.scrollY > 10)
    window.addEventListener('scroll', handler)
    return () => window.removeEventListener('scroll', handler)
  }, [])

  const links = [
    { to: '/', label: 'Home' },
    { to: '/predict', label: 'Risk Assessment' },
    { to: '/about', label: 'About' },
  ]

  return (
    <nav style={{
      position: 'fixed', top: 0, left: 0, right: 0, zIndex: 100,
      background: scrolled ? 'rgba(255,255,255,0.95)' : 'rgba(255,255,255,0.9)',
      backdropFilter: 'blur(12px)',
      borderBottom: scrolled ? '1px solid #e5e7eb' : '1px solid transparent',
      transition: 'all 0.3s ease',
      boxShadow: scrolled ? '0 2px 12px rgba(0,0,0,0.06)' : 'none',
    }}>
      <div style={{
        maxWidth: 1200, margin: '0 auto',
        padding: '0 24px',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        height: 68,
      }}>
        {/* Logo */}
        <Link to="/" style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <HeartIcon />
          <div>
            <span style={{ fontWeight: 800, fontSize: 17, color: '#111827', letterSpacing: '-0.3px' }}>
              CVD<span style={{ color: '#22c55e' }}>Risk</span>
            </span>
            <div style={{ fontSize: 10, color: '#6b7280', marginTop: -2, letterSpacing: '0.5px' }}>
              AI DETECTION
            </div>
          </div>
        </Link>

        {/* Nav links */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          {links.map(({ to, label }) => {
            const active = pathname === to
            return (
              <Link key={to} to={to} style={{
                padding: '8px 16px',
                borderRadius: 8,
                fontSize: 14,
                fontWeight: active ? 600 : 500,
                color: active ? '#16a34a' : '#374151',
                background: active ? '#f0fdf4' : 'transparent',
                transition: 'all 0.2s',
              }}
              onMouseEnter={e => { if (!active) e.target.style.background = '#f9fafb' }}
              onMouseLeave={e => { if (!active) e.target.style.background = 'transparent' }}
              >
                {label}
              </Link>
            )
          })}
          <Link to="/predict" style={{
            marginLeft: 8,
            padding: '9px 20px',
            borderRadius: 10,
            fontSize: 14,
            fontWeight: 600,
            color: '#fff',
            background: 'linear-gradient(135deg, #22c55e, #16a34a)',
            boxShadow: '0 2px 8px rgba(34,197,94,0.3)',
            transition: 'all 0.2s',
          }}
          onMouseEnter={e => { e.currentTarget.style.transform = 'translateY(-1px)'; e.currentTarget.style.boxShadow = '0 4px 12px rgba(34,197,94,0.4)' }}
          onMouseLeave={e => { e.currentTarget.style.transform = 'translateY(0)'; e.currentTarget.style.boxShadow = '0 2px 8px rgba(34,197,94,0.3)' }}
          >
            Check My Risk
          </Link>
        </div>
      </div>
    </nav>
  )
}
