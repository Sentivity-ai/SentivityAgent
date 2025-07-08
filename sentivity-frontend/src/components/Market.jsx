import React, { useEffect, useState } from 'react';
import { Box, Grid, Paper, Typography, List, ListItem, ListItemAvatar, ListItemText, Avatar, Skeleton, Tooltip } from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as ReTooltip, ResponsiveContainer } from 'recharts';
import axios from 'axios';

export default function Market() {
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

  return (
    <Box sx={{ width: '100vw', minHeight: '100vh', bgcolor: '#181c27', p: { xs: 1, md: 4 }, boxSizing: 'border-box' }}>
      <Typography variant="h4" color="#fff" fontWeight={700} mb={3} sx={{ letterSpacing: 1 }}>Market Overview</Typography>
      <Grid container spacing={4}>
        <Grid item xs={12} md={7}>
          <Paper sx={{ bgcolor: 'linear-gradient(135deg, #23293a 60%, #42a5f5 100%)', borderRadius: 4, boxShadow: 8, p: 3, mb: 4, transition: '0.3s', '&:hover': { boxShadow: '0 0 24px #42a5f5aa', transform: 'scale(1.03)' } }}>
            <Typography variant="h6" color="#fff" mb={2}>Market Summary</Typography>
            {loading ? (
              <Skeleton variant="rectangular" width="100%" height={120} sx={{ bgcolor: '#23293a' }} />
            ) : (
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
                        <Tooltip title={`View details for ${ticker}`} arrow>
                          <td style={{ padding: 10, fontWeight: 700 }}>{ticker}</td>
                        </Tooltip>
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
            )}
          </Paper>
          <Paper sx={{ bgcolor: 'linear-gradient(135deg, #23293a 60%, #42a5f5 100%)', borderRadius: 4, boxShadow: 8, p: 3, transition: '0.3s', '&:hover': { boxShadow: '0 0 24px #42a5f5aa', transform: 'scale(1.03)' } }}>
            <Typography variant="h6" color="#fff" mb={2}>Sector Trends</Typography>
            {loading ? (
              <Skeleton variant="rectangular" width="100%" height={180} sx={{ bgcolor: '#23293a' }} />
            ) : (
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={data.sector_trends} margin={{ top: 20, right: 30, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#23293a" />
                  <XAxis dataKey="sector" stroke="#aaa" />
                  <YAxis stroke="#aaa" />
                  <ReTooltip contentStyle={{ background: '#23293a', border: 'none', color: '#fff' }} />
                  <Bar dataKey="avg_sentiment" fill="#42a5f5" name="Avg Sentiment" radius={[8,8,0,0]} isAnimationActive />
                </BarChart>
              </ResponsiveContainer>
            )}
          </Paper>
        </Grid>
        <Grid item xs={12} md={5}>
          <Paper sx={{ bgcolor: 'linear-gradient(135deg, #23293a 60%, #66bb6a 100%)', borderRadius: 4, boxShadow: 8, p: 3, minHeight: 300, transition: '0.3s', '&:hover': { boxShadow: '0 0 24px #66bb6aaa', transform: 'scale(1.03)' } }}>
            <Typography variant="h6" color="#fff" mb={2}>Top Movers</Typography>
            {loading ? (
              <Skeleton variant="rectangular" width="100%" height={180} sx={{ bgcolor: '#23293a' }} />
            ) : (
              <List>
                {data.top_movers.map((m, i) => (
                  <Tooltip title={`Change: ${m.change > 0 ? '+' : ''}${m.change}`} arrow key={i}>
                    <ListItem sx={{ mb: 1, borderRadius: 2, transition: 'background 0.2s', '&:hover': { background: '#23293a' } }}>
                      <ListItemAvatar>
                        <Avatar sx={{ bgcolor: m.change > 0 ? '#66bb6a' : '#ff5252', color: '#fff', fontWeight: 700, fontSize: 18, boxShadow: m.change > 0 ? '0 0 8px #66bb6a55' : '0 0 8px #ff525255' }}>{m.ticker[0]}</Avatar>
                      </ListItemAvatar>
                      <ListItemText
                        primary={<span style={{ fontWeight: 700, fontSize: 18 }}>{m.ticker}</span>}
                        secondary={<span style={{ color: m.change > 0 ? '#66bb6a' : '#ff5252', fontWeight: 700, fontSize: 16 }}>{m.change > 0 ? '+' : ''}{m.change}</span>}
                      />
                    </ListItem>
                  </Tooltip>
                ))}
              </List>
            )}
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
} 