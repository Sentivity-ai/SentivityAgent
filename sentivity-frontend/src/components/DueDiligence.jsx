import React, { useState } from 'react';
import { Box, Paper, Typography, TextField, Button, Alert, Tooltip, Skeleton } from '@mui/material';
import axios from 'axios';

export default function DueDiligence() {
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
    <Box sx={{ width: '100vw', minHeight: '100vh', bgcolor: '#181c27', p: { xs: 1, md: 4 }, boxSizing: 'border-box' }}>
      <Typography variant="h4" color="#fff" fontWeight={700} mb={3} sx={{ letterSpacing: 1 }}>Due Diligence Agent</Typography>
      <Paper sx={{ p: 4, mb: 3, bgcolor: 'linear-gradient(135deg, #23293a 60%, #ff5252 100%)', borderRadius: 4, boxShadow: 8, transition: '0.3s', '&:hover': { boxShadow: '0 0 24px #ff5252aa', transform: 'scale(1.03)' } }}>
        <form onSubmit={handleSubmit} style={{ display: 'flex', gap: 16, flexWrap: 'wrap', alignItems: 'center' }}>
          <Tooltip title="Enter a stock ticker (e.g., AAPL)" arrow>
            <TextField label="Ticker" value={ticker} onChange={e => setTicker(e.target.value)} required sx={{ input: { color: '#fff' }, label: { color: '#aaa' } }} InputLabelProps={{ style: { color: '#aaa' } }} />
          </Tooltip>
          <Tooltip title="Start date for analysis (optional)" arrow>
            <TextField label="Start Date" type="date" value={startDate} onChange={e => setStartDate(e.target.value)} InputLabelProps={{ shrink: true, style: { color: '#aaa' } }} sx={{ input: { color: '#fff' } }} />
          </Tooltip>
          <Tooltip title="End date for analysis (optional)" arrow>
            <TextField label="End Date" type="date" value={endDate} onChange={e => setEndDate(e.target.value)} InputLabelProps={{ shrink: true, style: { color: '#aaa' } }} sx={{ input: { color: '#fff' } }} />
          </Tooltip>
          <Button type="submit" variant="contained" color="error" disabled={loading} sx={{ minWidth: 160, fontWeight: 700 }}>
            {loading ? 'Generating...' : 'Run Due Diligence'}
          </Button>
        </form>
        {error && <Alert severity="error" sx={{ mt: 2 }}>{error}</Alert>}
      </Paper>
      {loading && <Skeleton variant="rectangular" width="100%" height={180} sx={{ bgcolor: '#23293a', borderRadius: 4, mb: 3 }} />}
      {result && (
        <Paper sx={{ p: 4, bgcolor: 'linear-gradient(135deg, #23293a 60%, #66bb6a 100%)', color: '#fff', borderRadius: 4, boxShadow: 8, transition: '0.3s', '&:hover': { boxShadow: '0 0 24px #66bb6aaa', transform: 'scale(1.03)' } }}>
          <Typography variant="h5" fontWeight={700} mb={1}>Report for {result.ticker}</Typography>
          <Tooltip title="Risk level based on sentiment and analysis" arrow>
            <Typography variant="subtitle1" color="#ff5252" fontWeight={700} mb={2}>Risk Level: {result.risk_level}</Typography>
          </Tooltip>
          <Tooltip title="Model recommendation" arrow>
            <Typography variant="subtitle2" color="#aaa" mb={2}>Recommendation: {result.recommendation}</Typography>
          </Tooltip>
          <Box sx={{ whiteSpace: 'pre-wrap', fontFamily: 'monospace', fontSize: 16 }}>{result.report}</Box>
        </Paper>
      )}
    </Box>
  );
} 