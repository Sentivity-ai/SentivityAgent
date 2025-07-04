import React from 'react';
import { Drawer, List, ListItem, ListItemIcon, ListItemText, Toolbar, Typography, Box } from '@mui/material';
import DashboardIcon from '@mui/icons-material/Dashboard';
import BarChartIcon from '@mui/icons-material/BarChart';
import ChatIcon from '@mui/icons-material/Chat';
import ShowChartIcon from '@mui/icons-material/ShowChart';
import SettingsIcon from '@mui/icons-material/Settings';
import FactCheckIcon from '@mui/icons-material/FactCheck';
import { NavLink } from 'react-router-dom';

const drawerWidth = 220;

const navItems = [
  { text: 'Dashboard', icon: <DashboardIcon />, path: '/' },
  { text: 'Analytics', icon: <BarChartIcon />, path: '/analytics' },
  { text: 'Due Diligence', icon: <FactCheckIcon sx={{ color: '#ff5252' }} />, path: '/due-diligence' },
  { text: 'Market', icon: <ShowChartIcon />, path: '/market' },
  { text: 'Settings', icon: <SettingsIcon />, path: '/settings' },
];

export default function Sidebar() {
  return (
    <Drawer
      variant="permanent"
      sx={{
        width: drawerWidth,
        flexShrink: 0,
        '& .MuiDrawer-paper': {
          width: drawerWidth,
          boxSizing: 'border-box',
          bgcolor: '#1a1d2b',
          color: '#fff',
          backdropFilter: 'blur(10px)',
          background: 'rgba(26, 29, 43, 0.8)',
        },
      }}
    >
      <Toolbar>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <img src="/logo192.png" alt="Sentivity Logo" style={{ width: 32, height: 32 }} />
          <Typography variant="h6" fontWeight={700} color="#ff5252">Sentivity</Typography>
        </Box>
      </Toolbar>
      <List>
        {navItems.map((item) => (
          <NavLink
            to={item.path}
            key={item.text}
            style={({ isActive }) => ({
              textDecoration: 'none',
              color: isActive ? '#ff5252' : '#fff',
              background: isActive ? 'rgba(255,82,82,0.08)' : 'none',
              display: 'block',
              transition: 'all 0.3s ease-in-out',
              '&:hover': {
                transform: 'scale(1.02)',
                background: 'rgba(255,82,82,0.15)',
              },
            })}
            end={item.path === '/'}
          >
            <ListItem button selected={window.location.pathname === item.path}>
              <ListItemIcon sx={{ color: 'inherit' }}>{item.icon}</ListItemIcon>
              <ListItemText primary={item.text} />
            </ListItem>
          </NavLink>
        ))}
      </List>
    </Drawer>
  );
} 