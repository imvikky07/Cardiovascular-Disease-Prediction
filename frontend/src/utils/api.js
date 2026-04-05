import axios from 'axios'

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 15000,
  headers: { 'Content-Type': 'application/json' },
})

api.interceptors.response.use(
  (res) => res,
  (err) => {
    if (err.response) {
      const msg =
        err.response.data?.detail ||
        err.response.data?.message ||
        `Server error (${err.response.status})`
      return Promise.reject(new Error(msg))
    }
    if (err.code === 'ECONNABORTED') {
      return Promise.reject(new Error('Request timed out. Please try again.'))
    }
    return Promise.reject(new Error('Unable to reach the server. Is the backend running?'))
  }
)

export const predictCVD = async (formData) => {
  const payload = {
    age: Number(formData.age),
    gender: formData.gender,
    cholesterol: Number(formData.cholesterol),
    blood_pressure: Number(formData.blood_pressure),
    glucose: Number(formData.glucose),
    smoking: Number(formData.smoking),
    alcohol: Number(formData.alcohol),
    bmi: parseFloat(formData.bmi),
    physical_activity: Number(formData.physical_activity),
  }
  const res = await api.post('/api/predict', payload)
  return res.data
}

export const checkHealth = async () => {
  const res = await api.get('/health')
  return res.data
}

export default api
