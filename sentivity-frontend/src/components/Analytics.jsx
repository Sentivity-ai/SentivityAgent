import React, { useEffect, useState } from 'react';
import { Box, Grid, Paper, Typography, Tooltip, Skeleton } from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as ReTooltip, ResponsiveContainer, LineChart, Line } from 'recharts';
import axios from 'axios';

export default function Analytics() {
  const [stats, setStats] = useState(null);
  const [sentimentOverTime, setSentimentOverTime] = useState([]);
  const [categories, setCategories] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function fetchAnalytics() {
      setLoading(true);
      setError(null);
      try {
        const res = await axios.get('http://localhost:8000/posts/');
        const posts = res.data;
        // Calculate stats
        const totalPosts = posts.length;
        const sentimentScores = posts.map(p => typeof p.sentiment_score === 'number' ? p.sentiment_score : 0);
        const avgSentiment = sentimentScores.length ? (sentimentScores.reduce((sum, s) => sum + s, 0) / sentimentScores.length).toFixed(2) : 0;
        // Sentiment over time (group by date)
        const sentimentByDate = {};
        posts.forEach(p => {
          const date = p.created_utc ? new Date(p.created_utc).toISOString().slice(0, 10) : 'Unknown';
          if (!sentimentByDate[date]) sentimentByDate[date] = { date, sentiment: 0, count: 0 };
          sentimentByDate[date].sentiment += typeof p.sentiment_score === 'number' ? p.sentiment_score : 0;
          sentimentByDate[date].count += 1;
        });
        const sentimentOverTimeArr = Object.values(sentimentByDate).map(d => ({
          date: d.date,
          avgSentiment: d.count ? (d.sentiment / d.count).toFixed(2) : 0
        })).sort((a, b) => a.date.localeCompare(b.date));
        // Categories
        const freq = {};
        posts.forEach((p) => {
          const cat = p.ticker || p.subreddit || 'UNKNOWN';
          freq[cat] = (freq[cat] || 0) + 1;
        });
        const cats = Object.entries(freq)
          .map(([category, totalFreq]) => ({ category, totalFreq }))
          .sort((a, b) => b.totalFreq - a.totalFreq)
          .slice(0, 10);
        // Top category
        let topCategory = '-';
        if (cats && cats.length) {
          topCategory = cats[0].category;
        }
        setStats({ totalPosts, avgSentiment, topCategory });
        setSentimentOverTime(sentimentOverTimeArr);
        setCategories(cats);
      } catch (err) {
        setError(err.response?.data?.detail || err.message || 'Failed to load analytics');
      } finally {
        setLoading(false);
      }
    }
    fetchAnalytics();
  }, []);

  return (
    <Box sx={{ width: '100vw', minHeight: '100vh', bgcolor: '#181c27', p: { xs: 1, md: 4 }, boxSizing: 'border-box' }}>
      <Typography variant="h4" color="#fff" fontWeight={700} mb={3} sx={{ letterSpacing: 1 }}>Analytics</Typography>
      <Grid container spacing={4}>
        <Grid item xs={12} md={4}>
          <Paper sx={{ bgcolor: 'linear-gradient(135deg, #23293a 60%, #42a5f5 100%)', borderRadius: 4, boxShadow: 8, p: 3, mb: 4, transition: '0.3s', '&:hover': { boxShadow: '0 0 24px #42a5f5aa', transform: 'scale(1.03)' } }}>
            <Typography variant="h6" color="#fff" mb={2}>Key Metrics</Typography>
            {loading ? (
              <Skeleton variant="rectangular" width={180} height={80} sx={{ bgcolor: '#23293a' }} />
            ) : (
              <Box>
                <Tooltip title="Total number of posts analyzed" arrow><Typography color="#aaa" mb={1}>Total Posts: <b style={{ color: '#42a5f5' }}>{stats.totalPosts}</b></Typography></Tooltip>
                <Tooltip title="Average sentiment score (-1 to 1)" arrow><Typography color="#aaa" mb={1}>Avg. Sentiment: <b style={{ color: '#66bb6a' }}>{stats.avgSentiment}</b></Typography></Tooltip>
                <Tooltip title="Most mentioned ticker or category" arrow><Typography color="#aaa" mb={1}>Top Category: <b style={{ color: '#ffb300' }}>{stats.topCategory}</b></Typography></Tooltip>
              </Box>
            )}
          </Paper>
        </Grid>
        <Grid item xs={12} md={8}>
          <Paper sx={{ bgcolor: 'linear-gradient(135deg, #23293a 60%, #42a5f5 100%)', borderRadius: 4, boxShadow: 8, p: 3, minHeight: 220, transition: '0.3s', '&:hover': { boxShadow: '0 0 24px #42a5f5aa', transform: 'scale(1.03)' } }}>
            <Typography variant="h6" color="#fff" mb={2}>Sentiment Over Time</Typography>
            {loading ? (
              <Skeleton variant="rectangular" width="100%" height={180} sx={{ bgcolor: '#23293a' }} />
            ) : (
              <ResponsiveContainer width="100%" height={180}>
                <LineChart data={sentimentOverTime} margin={{ top: 20, right: 30, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#23293a" />
                  <XAxis dataKey="date" stroke="#aaa" />
                  <YAxis stroke="#aaa" />
                  <ReTooltip contentStyle={{ background: '#23293a', border: 'none', color: '#fff' }} />
                  <Line type="monotone" dataKey="avgSentiment" stroke="#42a5f5" strokeWidth={3} dot={false} isAnimationActive />
                </LineChart>
              </ResponsiveContainer>
            )}
          </Paper>
        </Grid>
        <Grid item xs={12}>
          <Paper sx={{ bgcolor: 'linear-gradient(135deg, #23293a 60%, #ffb300 100%)', borderRadius: 4, boxShadow: 8, p: 3, transition: '0.3s', '&:hover': { boxShadow: '0 0 24px #ffb300aa', transform: 'scale(1.03)' } }}>
            <Typography variant="h6" color="#fff" mb={2}>Trending Categories</Typography>
            {loading ? (
              <Skeleton variant="rectangular" width="100%" height={120} sx={{ bgcolor: '#23293a' }} />
            ) : (
              <ResponsiveContainer width="100%" height={120}>
                <BarChart data={categories} margin={{ top: 20, right: 30, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#23293a" />
                  <XAxis dataKey="category" stroke="#aaa" />
                  <YAxis stroke="#aaa" />
                  <ReTooltip contentStyle={{ background: '#23293a', border: 'none', color: '#fff' }} />
                  <Bar dataKey="totalFreq" fill="#ffb300" name="Mentions" radius={[8,8,0,0]} isAnimationActive />
                </BarChart>
              </ResponsiveContainer>
            )}
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
} 