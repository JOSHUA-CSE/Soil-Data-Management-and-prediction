import axios from 'axios';

const api = axios.create({
  baseURL: 'http://localhost:8000/api',
  headers: {
    'Content-Type': 'application/json',
  },
});

api.interceptors.response.use(
  (response) => response,
  (error) => {
    const errorMessage = error.response?.data?.error || error.message || 'An error occurred';
    return Promise.reject({ message: errorMessage, response: error.response });
  }
);

export const predictAPI = (data) => api.post('/predict', data);
export const historyAPI = () => api.get('/history');

export default api;
