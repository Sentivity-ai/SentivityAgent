import React, { useState } from 'react';
import { Box, Paper, Typography, TextField, Button, Alert } from '@mui/material';
import { supabase } from '../api/sentivity';

export default function Login({ onLogin }) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleLogin = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    const { error } = await supabase.auth.signInWithPassword({ email, password });
    setLoading(false);
    if (error) setError(error.message);
    else if (onLogin) onLogin();
  };

  return (
    <Box sx={{ minHeight: '100vh', bgcolor: '#181c27', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <Paper sx={{ p: 4, borderRadius: 4, boxShadow: 8, minWidth: 350, bgcolor: '#23293a' }}>
        <Typography variant="h4" color="#ff5252" fontWeight={700} mb={2} align="center">Sentivity Login</Typography>
        <form onSubmit={handleLogin}>
          <TextField
            label="Email"
            type="email"
            value={email}
            onChange={e => setEmail(e.target.value)}
            fullWidth
            required
            sx={{ mb: 2, input: { color: '#fff' }, label: { color: '#aaa' } }}
            InputLabelProps={{ style: { color: '#aaa' } }}
          />
          <TextField
            label="Password"
            type="password"
            value={password}
            onChange={e => setPassword(e.target.value)}
            fullWidth
            required
            sx={{ mb: 2, input: { color: '#fff' }, label: { color: '#aaa' } }}
            InputLabelProps={{ style: { color: '#aaa' } }}
          />
          {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
          <Button type="submit" variant="contained" color="error" fullWidth disabled={loading} sx={{ fontWeight: 700, py: 1.2 }}>
            {loading ? 'Logging in...' : 'Login'}
          </Button>
        </form>
      </Paper>
    </Box>
  );
} 