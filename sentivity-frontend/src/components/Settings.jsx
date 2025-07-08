import React, { useState } from 'react';
import { Box, Paper, Typography, Button, Switch, Tooltip, Grid } from '@mui/material';
import { supabase } from '../api/sentivity';

export default function Settings() {
  const [darkMode, setDarkMode] = useState(true);
  const [notifications, setNotifications] = useState(true);
  const [beta, setBeta] = useState(false);
  const user = supabase.auth.user();

  const handleLogout = async () => {
    await supabase.auth.signOut();
    window.location.reload();
  };

  return (
    <Box sx={{ width: '100vw', minHeight: '100vh', bgcolor: '#181c27', p: { xs: 1, md: 4 }, boxSizing: 'border-box' }}>
      <Typography variant="h4" color="#fff" fontWeight={700} mb={3} sx={{ letterSpacing: 1 }}>Settings</Typography>
      <Grid container spacing={4}>
        <Grid item xs={12} md={6}>
          <Paper sx={{ bgcolor: 'linear-gradient(135deg, #23293a 60%, #42a5f5 100%)', borderRadius: 4, boxShadow: 8, p: 3, transition: '0.3s', '&:hover': { boxShadow: '0 0 24px #42a5f5aa', transform: 'scale(1.03)' } }}>
            <Typography variant="h6" color="#fff" mb={2}>User Preferences</Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Tooltip title="Toggle dark mode" arrow>
                  <Typography color="#aaa">Dark Mode</Typography>
                </Tooltip>
                <Switch checked={darkMode} onChange={() => setDarkMode(v => !v)} color="error" />
              </Box>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Tooltip title="Enable/disable notifications" arrow>
                  <Typography color="#aaa">Notifications</Typography>
                </Tooltip>
                <Switch checked={notifications} onChange={() => setNotifications(v => !v)} color="error" />
              </Box>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Tooltip title="Enable beta features" arrow>
                  <Typography color="#aaa">Beta Features</Typography>
                </Tooltip>
                <Switch checked={beta} onChange={() => setBeta(v => !v)} color="error" />
              </Box>
            </Box>
          </Paper>
        </Grid>
        <Grid item xs={12} md={6}>
          <Paper sx={{ bgcolor: 'linear-gradient(135deg, #23293a 60%, #ff5252 100%)', borderRadius: 4, boxShadow: 8, p: 3, transition: '0.3s', '&:hover': { boxShadow: '0 0 24px #ff5252aa', transform: 'scale(1.03)' } }}>
            <Typography variant="h6" color="#fff" mb={2}>Account</Typography>
            <Typography color="#aaa" mb={2}>Email: {user?.email || 'Unknown'}</Typography>
            <Tooltip title="Logout from Sentivity" arrow>
              <Button variant="contained" color="error" onClick={handleLogout} sx={{ fontWeight: 700 }}>
                Logout
              </Button>
            </Tooltip>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
} 