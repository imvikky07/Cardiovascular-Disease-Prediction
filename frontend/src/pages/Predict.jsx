import React, { useState } from 'react'
import { useForm } from 'react-hook-form'
import { usePrediction } from '../hooks/usePrediction'
import ResultCard from '../components/ResultCard'

const Field = ({ label, error, hint, children }) => (
  <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
    <label style={{ fontSize: 13, fontWeight: 600, color: '#374151' }}>
      {label}
      {hint && <span style={{ fontWeight: 400, color: '#9ca3af', marginLeft: 6 }}>({hint})</span>}
    </label>
    {children}
    {error && (
      <span style={{ fontSize: 12, color: '#ef4444', display: 'flex', alignItems: 'center', gap: 4 }}>
        ⚠ {error}
      </span>
    )}
  </div>
)

const inputStyle = (hasError) => ({
  width: '100%',
  padding: '11px 14px',
  borderRadius: 10,
  border: `1.5px solid ${hasError ? '#fca5a5' : '#d1d5db'}`,
  fontSize: 14,
  color: '#111827',
  background: hasError ? '#fef2f2' : '#fff',
  transition: 'all 0.2s',
  appearance: 'none',
  WebkitAppearance: 'none',
})

const ToggleGroup = ({ value, onChange, options }) => (
  <div style={{ display: 'flex', gap: 8 }}>
    {options.map(opt => (
      <button
        key={opt.value}
        type="button"
        onClick={() => onChange(opt.value)}
        style={{
          flex: 1, padding: '10px 0', borderRadius: 10,
          fontSize: 13, fontWeight: 600,
          border: `2px solid ${value === opt.value ? '#22c55e' : '#e5e7eb'}`,
          background: value === opt.value ? '#f0fdf4' : '#fff',
          color: value === opt.value ? '#16a34a' : '#6b7280',
          cursor: 'pointer', transition: 'all 0.2s',
        }}
      >
        {opt.label}
      </button>
    ))}
  </div>
)

const FIELDS_CONFIG = [
  {
    id: 'age', label: 'Age', hint: '1–120 years', type: 'number',
    validation: { required: 'Age is required', min: { value: 1, message: 'Min 1' }, max: { value: 120, message: 'Max 120' } },
    placeholder: 'e.g. 55',
  },
  {
    id: 'cholesterol', label: 'Total Cholesterol', hint: 'mg/dL', type: 'number',
    validation: { required: 'Required', min: { value: 100, message: 'Min 100' }, max: { value: 400, message: 'Max 400' } },
    placeholder: 'e.g. 220',
  },
  {
    id: 'blood_pressure', label: 'Systolic Blood Pressure', hint: 'mmHg', type: 'number',
    validation: { required: 'Required', min: { value: 50, message: 'Min 50' }, max: { value: 250, message: 'Max 250' } },
    placeholder: 'e.g. 130',
  },
  {
    id: 'glucose', label: 'Fasting Glucose', hint: 'mg/dL', type: 'number',
    validation: { required: 'Required', min: { value: 50, message: 'Min 50' }, max: { value: 400, message: 'Max 400' } },
    placeholder: 'e.g. 95',
  },
  {
    id: 'bmi', label: 'BMI', hint: 'Body Mass Index', type: 'number', step: '0.1',
    validation: { required: 'Required', min: { value: 10, message: 'Min 10' }, max: { value: 60, message: 'Max 60' } },
    placeholder: 'e.g. 25.4',
  },
]

export default function Predict() {
  const { register, handleSubmit, setValue, watch, formState: { errors } } = useForm({
    defaultValues: { gender: 'Male', smoking: 0, alcohol: 0, physical_activity: 1 },
  })
  const { result, loading, error, predict, reset } = usePrediction()
  const [submitted, setSubmitted] = useState(false)

  const gender = watch('gender')
  const smoking = watch('smoking')
  const alcohol = watch('alcohol')
  const physical_activity = watch('physical_activity')

  const onSubmit = async (data) => {
    setSubmitted(true)
    try {
      await predict(data)
      setTimeout(() => {
        document.getElementById('result-section')?.scrollIntoView({ behavior: 'smooth', block: 'start' })
      }, 100)
    } catch (_) {}
  }

  const handleReset = () => {
    reset()
    setSubmitted(false)
  }

  return (
    <div style={{ paddingTop: 68, minHeight: '100vh', background: '#f9fafb' }}>
      <div style={{ maxWidth: 760, margin: '0 auto', padding: '40px 24px 80px' }}>
        {/* Header */}
        <div style={{ textAlign: 'center', marginBottom: 36 }}>
          <div style={{
            display: 'inline-flex', alignItems: 'center', gap: 6,
            background: '#dcfce7', border: '1px solid #bbf7d0',
            borderRadius: 99, padding: '5px 14px',
            fontSize: 12, color: '#15803d', fontWeight: 600, marginBottom: 16,
          }}>
            🩺 Clinical Risk Assessment
          </div>
          <h1 style={{ fontSize: 32, fontWeight: 900, color: '#111827', letterSpacing: '-1px', marginBottom: 10 }}>
            CVD Risk Assessment
          </h1>
          <p style={{ color: '#6b7280', fontSize: 15, lineHeight: 1.6 }}>
            Complete all fields accurately for the most reliable prediction. All values are validated before submission.
          </p>
        </div>

        {/* Form Card */}
        <div style={{
          background: '#fff', borderRadius: 20,
          border: '1px solid #e5e7eb',
          boxShadow: '0 4px 16px rgba(0,0,0,0.06)',
          padding: '36px',
          marginBottom: 28,
        }}>
          <form onSubmit={handleSubmit(onSubmit)} noValidate>
            {/* Section: Demographics */}
            <div style={{ marginBottom: 28 }}>
              <div style={{
                fontSize: 11, fontWeight: 700, color: '#22c55e',
                textTransform: 'uppercase', letterSpacing: '1px', marginBottom: 16,
                paddingBottom: 10, borderBottom: '1px solid #f0fdf4',
              }}>
                Demographics
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 18 }}>
                <Field label="Age" hint="1–120 yrs" error={errors.age?.message}>
                  <input
                    type="number"
                    placeholder="e.g. 55"
                    style={inputStyle(!!errors.age)}
                    {...register('age', { required: 'Age is required', min: { value: 1, message: 'Min 1' }, max: { value: 120, message: 'Max 120' } })}
                    onFocus={e => { if (!errors.age) e.target.style.borderColor = '#22c55e' }}
                    onBlur={e => { if (!errors.age) e.target.style.borderColor = '#d1d5db' }}
                  />
                </Field>
                <Field label="Gender" error={errors.gender?.message}>
                  <ToggleGroup
                    value={gender}
                    onChange={(v) => setValue('gender', v)}
                    options={[{ value: 'Male', label: '♂ Male' }, { value: 'Female', label: '♀ Female' }]}
                  />
                  <input type="hidden" {...register('gender', { required: true })} />
                </Field>
              </div>
            </div>

            {/* Section: Clinical Measurements */}
            <div style={{ marginBottom: 28 }}>
              <div style={{
                fontSize: 11, fontWeight: 700, color: '#22c55e',
                textTransform: 'uppercase', letterSpacing: '1px', marginBottom: 16,
                paddingBottom: 10, borderBottom: '1px solid #f0fdf4',
              }}>
                Clinical Measurements
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 18 }}>
                {FIELDS_CONFIG.map(({ id, label, hint, type, step, validation, placeholder }) => (
                  <Field key={id} label={label} hint={hint} error={errors[id]?.message}>
                    <input
                      type={type}
                      step={step}
                      placeholder={placeholder}
                      style={inputStyle(!!errors[id])}
                      {...register(id, validation)}
                      onFocus={e => { if (!errors[id]) e.target.style.borderColor = '#22c55e' }}
                      onBlur={e => { if (!errors[id]) e.target.style.borderColor = '#d1d5db' }}
                    />
                  </Field>
                ))}
              </div>
            </div>

            {/* Section: Lifestyle */}
            <div style={{ marginBottom: 32 }}>
              <div style={{
                fontSize: 11, fontWeight: 700, color: '#22c55e',
                textTransform: 'uppercase', letterSpacing: '1px', marginBottom: 16,
                paddingBottom: 10, borderBottom: '1px solid #f0fdf4',
              }}>
                Lifestyle Factors
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 18 }}>
                <Field label="Smoking Status">
                  <ToggleGroup
                    value={smoking}
                    onChange={(v) => setValue('smoking', v)}
                    options={[{ value: 0, label: '🚭 No' }, { value: 1, label: '🚬 Yes' }]}
                  />
                  <input type="hidden" {...register('smoking')} />
                </Field>
                <Field label="Alcohol Intake">
                  <ToggleGroup
                    value={alcohol}
                    onChange={(v) => setValue('alcohol', v)}
                    options={[{ value: 0, label: '🚫 No' }, { value: 1, label: '🍺 Yes' }]}
                  />
                  <input type="hidden" {...register('alcohol')} />
                </Field>
                <Field label="Physically Active">
                  <ToggleGroup
                    value={physical_activity}
                    onChange={(v) => setValue('physical_activity', v)}
                    options={[{ value: 0, label: '🛋 No' }, { value: 1, label: '🏃 Yes' }]}
                  />
                  <input type="hidden" {...register('physical_activity')} />
                </Field>
              </div>
            </div>

            {/* Error Banner */}
            {error && (
              <div style={{
                background: '#fef2f2', border: '1px solid #fecaca',
                borderRadius: 10, padding: '14px 16px',
                color: '#dc2626', fontSize: 14, marginBottom: 20,
                display: 'flex', alignItems: 'center', gap: 8,
              }}>
                <span style={{ fontSize: 18 }}>⚠️</span>
                <span>{error}</span>
              </div>
            )}

            {/* Submit */}
            <button
              type="submit"
              disabled={loading}
              style={{
                width: '100%', padding: '15px',
                borderRadius: 12,
                background: loading ? '#d1fae5' : 'linear-gradient(135deg, #22c55e, #16a34a)',
                color: loading ? '#86efac' : '#fff',
                fontWeight: 700, fontSize: 16,
                cursor: loading ? 'not-allowed' : 'pointer',
                boxShadow: loading ? 'none' : '0 4px 14px rgba(34,197,94,0.35)',
                transition: 'all 0.25s',
                display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 10,
              }}
            >
              {loading ? (
                <>
                  <span style={{
                    width: 18, height: 18, border: '2.5px solid #86efac',
                    borderTopColor: '#16a34a', borderRadius: '50%',
                    display: 'inline-block',
                    animation: 'spin 0.9s linear infinite',
                  }}></span>
                  Analyzing risk factors...
                </>
              ) : (
                '🔍 Analyze CVD Risk'
              )}
            </button>
          </form>
        </div>

        {/* Disclaimer */}
        <div style={{
          background: '#fffbeb', border: '1px solid #fde68a',
          borderRadius: 12, padding: '14px 18px',
          fontSize: 13, color: '#92400e',
          display: 'flex', gap: 10, alignItems: 'flex-start',
          marginBottom: 28,
        }}>
          <span style={{ fontSize: 16, flexShrink: 0 }}>⚠️</span>
          <span>
            <strong>Medical Disclaimer:</strong> This tool is for educational purposes only and does not constitute medical advice. Always consult a qualified healthcare provider for diagnosis and treatment.
          </span>
        </div>

        {/* Result */}
        {submitted && (result || error) && (
          <div id="result-section" style={{ animation: 'fadeIn 0.45s ease' }}>
            {result && <ResultCard result={result} onReset={handleReset} />}
          </div>
        )}
      </div>
    </div>
  )
}
