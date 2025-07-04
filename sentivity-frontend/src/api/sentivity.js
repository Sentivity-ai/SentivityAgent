import axios from 'axios';

const API_BASE = 'http://localhost:8000';

export const getPosts = async (params = {}) => {
  const res = await axios.get(`${API_BASE}/posts/`, { params });
  return res.data;
};

export const getStats = async (ticker) => {
  const res = await axios.get(`${API_BASE}/posts/stats/${ticker}`);
  return res.data;
};

export const getTrendingWords = async () => {
  // For demo, fetch all posts and aggregate trending words client-side
  const posts = await getPosts({ limit: 500 });
  // Aggregate trending words (mock logic, replace with backend endpoint if available)
  const freq = {};
  posts.forEach((p) => {
    const words = (p.title + ' ' + p.body).toLowerCase().split(/\W+/);
    words.forEach((w) => {
      if (w.length > 3) freq[w] = (freq[w] || 0) + 1;
    });
  });
  const trending = Object.entries(freq)
    .map(([word, frequency]) => ({ word, frequency }))
    .sort((a, b) => b.frequency - a.frequency)
    .slice(0, 10);
  return trending;
};

export const getCategories = async () => {
  // Fetch all posts and aggregate categories client-side
  const posts = await getPosts({ limit: 500 });
  const freq = {};
  posts.forEach((p) => {
    const cat = p.ticker || p.subreddit || 'UNKNOWN';
    freq[cat] = (freq[cat] || 0) + 1;
  });
  return Object.entries(freq)
    .map(([category, totalFreq]) => ({ category, totalFreq }))
    .sort((a, b) => b.totalFreq - a.totalFreq)
    .slice(0, 10);
};

export const getSentiment = async (ticker) => {
  const res = await axios.post(`${API_BASE}/sentiment/predict`, { ticker });
  return res.data;
}; 