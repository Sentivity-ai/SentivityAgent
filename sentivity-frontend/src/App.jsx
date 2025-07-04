import React, { useState, useRef, useEffect } from 'react';
import { Box, CssBaseline, AppBar, Toolbar, Typography, Avatar, IconButton, Badge, TextField, Button, Paper, Alert, Grid, List, ListItem, ListItemAvatar, ListItemText } from '@mui/material';
import NotificationsIcon from '@mui/icons-material/Notifications';
import Sidebar from './components/Sidebar.jsx';
import Dashboard from './components/Dashboard.jsx';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import axios from 'axios';
import dayjs from 'dayjs';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line } from 'recharts';
import { getPosts, getCategories } from './api/sentivity.js';

const drawerWidth = 220;

function Analytics() {
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
        // Fetch posts for sentiment over time and stats
        const posts = await getPosts({ limit: 500 });
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

  if (loading) return <Typography color="#fff">Loading analytics...</Typography>;
  if (error) return <Alert severity="error">{error}</Alert>;

  return (
    <Box sx={{ width: '100vw', minHeight: '100vh', bgcolor: '#181c27', p: { xs: 1, md: 4 }, boxSizing: 'border-box' }}>
      <Typography variant="h4" color="#fff" fontWeight={700} mb={3} sx={{ letterSpacing: 1 }}>Analytics</Typography>
      <Grid container spacing={4}>
        <Grid item xs={12} md={4}>
          <Paper sx={{ bgcolor: '#181c27', borderRadius: 4, boxShadow: 8, p: 3, mb: 4 }}>
            <Typography variant="h6" color="#fff" mb={2}>Key Metrics</Typography>
            <Box>
              <Typography color="#aaa" mb={1}>Total Posts: <b style={{ color: '#42a5f5' }}>{stats.totalPosts}</b></Typography>
              <Typography color="#aaa" mb={1}>Avg. Sentiment: <b style={{ color: '#66bb6a' }}>{stats.avgSentiment}</b></Typography>
              <Typography color="#aaa" mb={1}>Top Category: <b style={{ color: '#ffb300' }}>{stats.topCategory}</b></Typography>
            </Box>
          </Paper>
        </Grid>
        <Grid item xs={12} md={8}>
          <Paper sx={{ bgcolor: '#181c27', borderRadius: 4, boxShadow: 8, p: 3, minHeight: 220 }}>
            <Typography variant="h6" color="#fff" mb={2}>Sentiment Over Time</Typography>
            <ResponsiveContainer width="100%" height={180}>
              <LineChart data={sentimentOverTime} margin={{ top: 20, right: 30, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#23293a" />
                <XAxis dataKey="date" stroke="#aaa" />
                <YAxis stroke="#aaa" />
                <Tooltip contentStyle={{ background: '#23293a', border: 'none', color: '#fff' }} />
                <Line type="monotone" dataKey="avgSentiment" stroke="#42a5f5" strokeWidth={3} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>
        <Grid item xs={12}>
          <Paper sx={{ bgcolor: '#181c27', borderRadius: 4, boxShadow: 8, p: 3 }}>
            <Typography variant="h6" color="#fff" mb={2}>Trending Categories</Typography>
            <ResponsiveContainer width="100%" height={120}>
              <BarChart data={categories} margin={{ top: 20, right: 30, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#23293a" />
                <XAxis dataKey="category" stroke="#aaa" />
                <YAxis stroke="#aaa" />
                <Tooltip contentStyle={{ background: '#23293a', border: 'none', color: '#fff' }} />
                <Bar dataKey="totalFreq" fill="#ffb300" name="Mentions" radius={[8,8,0,0]} />
              </BarChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
}
function Market() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function fetchMarket() {
      setLoading(true);
      setError(null);
      try {
        const res = await axios.get('http://localhost:8000/sentiment/market/overview');
        setData(res.data);
      } catch (err) {
        setError(err.response?.data?.detail || err.message || 'Failed to load market data');
      } finally {
        setLoading(false);
      }
    }
    fetchMarket();
  }, []);

  if (loading) return <Typography color="#fff">Loading market data...</Typography>;
  if (error) return <Alert severity="error">{error}</Alert>;

  return (
    <Box sx={{ width: '100vw', minHeight: '100vh', bgcolor: '#181c27', p: { xs: 1, md: 4 }, boxSizing: 'border-box' }}>
      <Typography variant="h4" color="#fff" fontWeight={700} mb={3} sx={{ letterSpacing: 1 }}>Market Overview</Typography>
      <Grid container spacing={4}>
        <Grid item xs={12} md={7}>
          <Paper sx={{ bgcolor: '#181c27', borderRadius: 4, boxShadow: 8, p: 3, mb: 4 }}>
            <Typography variant="h6" color="#fff" mb={2}>Market Summary</Typography>
            <Box sx={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', color: '#fff', borderCollapse: 'collapse', fontFamily: 'inherit' }}>
                <thead>
                  <tr style={{ background: 'linear-gradient(90deg, #23293a 60%, #1e2233 100%)' }}>
                    <th style={{ padding: 10, fontWeight: 700, letterSpacing: 1 }}>Ticker</th>
                    <th style={{ padding: 10, fontWeight: 700, letterSpacing: 1 }}>Price</th>
                    <th style={{ padding: 10, fontWeight: 700, letterSpacing: 1 }}>Change</th>
                    <th style={{ padding: 10, fontWeight: 700, letterSpacing: 1 }}>Volume</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(data.market_summary).map(([ticker, info]) => (
                    <tr key={ticker} style={{ borderBottom: '1px solid #23293a', transition: 'background 0.2s' }}
                      onMouseOver={e => e.currentTarget.style.background='#23293a'}
                      onMouseOut={e => e.currentTarget.style.background=''}
                    >
                      <td style={{ padding: 10, fontWeight: 700 }}>{ticker}</td>
                      <td style={{ padding: 10 }}>${info.price}</td>
                      <td style={{ padding: 10 }}>
                        <span style={{
                          background: info.change > 0 ? '#1e2e1e' : '#2d1e1e',
                          color: info.change > 0 ? '#66bb6a' : '#ff5252',
                          borderRadius: 8,
                          padding: '2px 10px',
                          fontWeight: 700,
                          fontSize: 15,
                          boxShadow: info.change > 0 ? '0 0 8px #66bb6a33' : '0 0 8px #ff525233',
                          border: info.change > 0 ? '1px solid #66bb6a' : '1px solid #ff5252',
                          display: 'inline-block',
                          minWidth: 60,
                          textAlign: 'center',
                        }}>{info.change > 0 ? '+' : ''}{info.change}</span>
                      </td>
                      <td style={{ padding: 10 }}>{info.volume.toLocaleString()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Box>
          </Paper>
          <Paper sx={{ bgcolor: '#181c27', borderRadius: 4, boxShadow: 8, p: 3 }}>
            <Typography variant="h6" color="#fff" mb={2}>Sector Trends</Typography>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={data.sector_trends} margin={{ top: 20, right: 30, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#23293a" />
                <XAxis dataKey="sector" stroke="#aaa" />
                <YAxis stroke="#aaa" />
                <Tooltip contentStyle={{ background: '#23293a', border: 'none', color: '#fff' }} />
                <Bar dataKey="avg_sentiment" fill="#42a5f5" name="Avg Sentiment" radius={[8,8,0,0]} />
              </BarChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>
        <Grid item xs={12} md={5}>
          <Paper sx={{ bgcolor: '#181c27', borderRadius: 4, boxShadow: 8, p: 3, minHeight: 300 }}>
            <Typography variant="h6" color="#fff" mb={2}>Top Movers</Typography>
            <List>
              {data.top_movers.map((m, i) => (
                <ListItem key={i} sx={{ mb: 1, borderRadius: 2, transition: 'background 0.2s', '&:hover': { background: '#23293a' } }}>
                  <ListItemAvatar>
                    <Avatar sx={{ bgcolor: m.change > 0 ? '#66bb6a' : '#ff5252', color: '#fff', fontWeight: 700, fontSize: 18, boxShadow: m.change > 0 ? '0 0 8px #66bb6a55' : '0 0 8px #ff525255' }}>{m.ticker[0]}</Avatar>
                  </ListItemAvatar>
                  <ListItemText
                    primary={<span style={{ fontWeight: 700, fontSize: 18 }}>{m.ticker}</span>}
                    secondary={<span style={{ color: m.change > 0 ? '#66bb6a' : '#ff5252', fontWeight: 700, fontSize: 16 }}>{m.change > 0 ? '+' : ''}{m.change}</span>}
                  />
                </ListItem>
              ))}
            </List>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
}
function Settings() {
  const [darkMode, setDarkMode] = useState(true);
  const [notifications, setNotifications] = useState(true);
  const [beta, setBeta] = useState(false);
  return (
    <Box sx={{ width: '100vw', minHeight: '100vh', bgcolor: '#181c27', p: { xs: 1, md: 4 }, boxSizing: 'border-box' }}>
      <Typography variant="h4" color="#fff" fontWeight={700} mb={3} sx={{ letterSpacing: 1 }}>Settings</Typography>
      <Grid container spacing={4}>
        <Grid item xs={12} md={6}>
          <Paper sx={{ bgcolor: '#181c27', borderRadius: 4, boxShadow: 8, p: 3 }}>
            <Typography variant="h6" color="#fff" mb={2}>User Preferences</Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Typography color="#aaa">Dark Mode</Typography>
                <Button variant={darkMode ? 'contained' : 'outlined'} color="error" onClick={() => setDarkMode(v => !v)} sx={{ minWidth: 100, fontWeight: 700 }}>
                  {darkMode ? 'On' : 'Off'}
                </Button>
              </Box>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Typography color="#aaa">Notifications</Typography>
                <Button variant={notifications ? 'contained' : 'outlined'} color="error" onClick={() => setNotifications(v => !v)} sx={{ minWidth: 100, fontWeight: 700 }}>
                  {notifications ? 'On' : 'Off'}
                </Button>
              </Box>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Typography color="#aaa">Beta Features</Typography>
                <Button variant={beta ? 'contained' : 'outlined'} color="error" onClick={() => setBeta(v => !v)} sx={{ minWidth: 100, fontWeight: 700 }}>
                  {beta ? 'Enabled' : 'Disabled'}
                </Button>
              </Box>
            </Box>
          </Paper>
        </Grid>
        <Grid item xs={12} md={6}>
          <Paper sx={{ bgcolor: '#181c27', borderRadius: 4, boxShadow: 8, p: 3 }}>
            <Typography variant="h6" color="#fff" mb={2}>Account</Typography>
            <Typography color="#aaa">Account management coming soon.</Typography>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
}

function DueDiligence() {
  const [ticker, setTicker] = useState('');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      if (!ticker) throw new Error('Ticker is required');
      const payload = {
        ticker: ticker.trim().toUpperCase(),
        start_date: startDate ? new Date(startDate).toISOString() : null,
        end_date: endDate ? new Date(endDate).toISOString() : null,
        include_due_diligence: true
      };
      const res = await axios.post('http://localhost:8000/sentiment/due-diligence', payload);
      setResult(res.data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to generate report');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box>
      <Typography variant="h4" color="#fff" fontWeight={700} mb={2}>Due Diligence Agent</Typography>
      <Paper sx={{ p: 3, mb: 3, bgcolor: '#23293a' }}>
        <form onSubmit={handleSubmit} style={{ display: 'flex', gap: 16, flexWrap: 'wrap', alignItems: 'center' }}>
          <TextField label="Ticker" value={ticker} onChange={e => setTicker(e.target.value)} required sx={{ input: { color: '#fff' }, label: { color: '#aaa' } }} InputLabelProps={{ style: { color: '#aaa' } }} />
          <TextField label="Start Date" type="date" value={startDate} onChange={e => setStartDate(e.target.value)} InputLabelProps={{ shrink: true, style: { color: '#aaa' } }} sx={{ input: { color: '#fff' } }} />
          <TextField label="End Date" type="date" value={endDate} onChange={e => setEndDate(e.target.value)} InputLabelProps={{ shrink: true, style: { color: '#aaa' } }} sx={{ input: { color: '#fff' } }} />
          <Button type="submit" variant="contained" color="error" disabled={loading} sx={{ minWidth: 160, fontWeight: 700 }}>
            {loading ? 'Generating...' : 'Run Due Diligence'}
          </Button>
        </form>
        {error && <Alert severity="error" sx={{ mt: 2 }}>{error}</Alert>}
      </Paper>
      {result && (
        <Paper sx={{ p: 3, bgcolor: '#181c27', color: '#fff' }}>
          <Typography variant="h5" fontWeight={700} mb={1}>Report for {result.ticker}</Typography>
          <Typography variant="subtitle1" color="#ff5252" fontWeight={700} mb={2}>Risk Level: {result.risk_level}</Typography>
          <Typography variant="subtitle2" color="#aaa" mb={2}>Recommendation: {result.recommendation}</Typography>
          <Box sx={{ whiteSpace: 'pre-wrap', fontFamily: 'monospace', fontSize: 16 }}>{result.report}</Box>
        </Paper>
      )}
    </Box>
  );
}

export default function App() {
  return (
    <Router>
      <Box sx={{ display: 'flex', bgcolor: '#181c27', minHeight: '100vh' }}>
        <CssBaseline />
        <Sidebar />
        <Box sx={{ flexGrow: 1, ml: `${drawerWidth}px` }}>
          <AppBar position="fixed" sx={{ ml: `${drawerWidth}px`, bgcolor: '#23293a', boxShadow: 'none', borderBottom: '1px solid #23293a' }}>
            <Toolbar>
              <Typography variant="h5" sx={{ flexGrow: 1, fontWeight: 700, color: '#ff5252' }}>
                Sentivity
              </Typography>
              <IconButton color="inherit" sx={{ mr: 2 }}>
                <Badge badgeContent={3} color="error">
                  <NotificationsIcon />
                </Badge>
              </IconButton>
              <Avatar alt="User" src="/avatar.png" sx={{ bgcolor: '#ff5252' }} />
            </Toolbar>
          </AppBar>
          <Toolbar />
          <Box sx={{ p: 4 }}>
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/analytics" element={<Analytics />} />
              <Route path="/market" element={<Market />} />
              <Route path="/settings" element={<Settings />} />
              <Route path="/due-diligence" element={<DueDiligence />} />
            </Routes>
          </Box>
        </Box>
      </Box>
    </Router>
  );
}
