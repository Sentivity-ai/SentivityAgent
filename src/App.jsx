import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { CssBaseline, Box, Toolbar, AppBar, Typography, IconButton } from '@mui/material';
import MenuIcon from '@mui/icons-material/Menu';
import Sidebar from './components/Sidebar';
import Dashboard from './components/Dashboard';

// Placeholder pages
function WordAnalytics() {
  return <Box p={3}><Typography variant="h4">Word Analytics</Typography></Box>;
}
function ChatAnalytics() {
  return <Box p={3}><Typography variant="h4">Chat with Analytics</Typography></Box>;
}
function MarketOverview() {
  return <Box p={3}><Typography variant="h4">Market Overview</Typography></Box>;
}

const drawerWidth = 240;

export default function App() {
  return (
    <Router>
      <Box sx={{ display: 'flex', minHeight: '100vh', background: '#11131a' }}>
        <CssBaseline />
        <Sidebar />
        <Box component="main" sx={{ flexGrow: 1, bgcolor: '#181c24', color: '#fff', minHeight: '100vh' }}>
          <AppBar
            position="fixed"
            sx={{
              width: `calc(100% - ${drawerWidth}px)`,
              ml: `${drawerWidth}px`,
              bgcolor: '#23293a',
              color: '#fff',
              boxShadow: 'none',
              borderBottom: '1px solid #23293a',
            }}
            elevation={0}
          >
            <Toolbar>
              <IconButton edge="start" color="inherit" aria-label="menu" sx={{ mr: 2, display: { sm: 'none' } }}>
                <MenuIcon />
              </IconButton>
              <Typography variant="h6" noWrap component="div">
                Sentivity B2B Secure Platform
              </Typography>
            </Toolbar>
          </AppBar>
          <Toolbar />
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/word-analytics" element={<WordAnalytics />} />
            <Route path="/chat" element={<ChatAnalytics />} />
            <Route path="/market-overview" element={<MarketOverview />} />
          </Routes>
        </Box>
      </Box>
    </Router>
  );
} 