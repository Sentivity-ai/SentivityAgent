import React, { useEffect, useState } from 'react';
import { Box, Grid, Card, CardContent, Typography, CircularProgress, Alert, Skeleton, Avatar, List, ListItem, ListItemAvatar, ListItemText, Paper } from '@mui/material';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import ForumIcon from '@mui/icons-material/Forum';
import EmojiEmotionsIcon from '@mui/icons-material/EmojiEmotions';
import CategoryIcon from '@mui/icons-material/Category';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line } from 'recharts';
import { getPosts, getTrendingWords, getCategories } from '../api/sentivity';
import axios from 'axios';

const statCardData = [
  { label: 'Total Posts', icon: <ForumIcon />, color: '#42a5f5' },
  { label: 'Trending Words', icon: <TrendingUpIcon />, color: '#ffb300' },
  { label: 'Avg. Sentiment', icon: <EmojiEmotionsIcon />, color: '#66bb6a' },
  { label: 'Categories', icon: <CategoryIcon />, color: '#ab47bc' },
];

const DEFAULT_TICKER = 'AAPL';

export default function Dashboard() {
  const [sentiment, setSentiment] = useState(null);
  const [trendingWords, setTrendingWords] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [posts, setPosts] = useState([]);

  useEffect(() => {
    async function fetchData() {
      setLoading(true);
      setError(null);
      try {
        const [statsRes, postsData, trending] = await Promise.all([
          axios.get(`http://localhost:8000/posts/stats/${DEFAULT_TICKER}`),
          getPosts({ limit: 500 }),
          getTrendingWords()
        ]);
        setStats(statsRes.data);
        setPosts(postsData);
        setTrendingWords(trending);
      } catch (err) {
        setError(err.response?.data?.detail || err.message || 'Failed to load analytics');
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, []);

  // Stat card values
  const statCardValues = stats ? [
    stats.total_posts || 0,
    trendingWords.length,
    stats.average_sentiment !== undefined ? stats.average_sentiment : '-',
    stats.top_subreddits ? stats.top_subreddits.length : 0
  ] : [0, 0, '-', 0];

  // Prepare activity feed (latest 5 posts)
  const activityFeed = posts.slice(0, 5);

  if (loading) return <Typography color="#fff">Loading analytics...</Typography>;
  if (error) return <Alert severity="error">{error}</Alert>;

  return (
    <Box>
      <Typography variant="h4" gutterBottom fontWeight={700} color="#fff">
        Dashboard Overview
      </Typography>
      <Grid container spacing={3} mb={3}>
        {statCardData.map((stat, i) => (
          <Grid item xs={12} sm={6} md={3} key={stat.label}>
            <Card sx={{ bgcolor: '#23293a', color: '#fff', borderRadius: 3, boxShadow: 3 }}>
              <CardContent sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                <Avatar sx={{ bgcolor: stat.color, width: 48, height: 48 }}>{stat.icon}</Avatar>
                <Box>
                  <Typography variant="subtitle2" color="#aaa" fontWeight={500}>{stat.label}</Typography>
                  <Typography variant="h4" fontWeight={700} color={stat.color}>{statCardValues[i]}</Typography>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          <Card sx={{ bgcolor: '#23293a', color: '#fff', borderRadius: 3, boxShadow: 3, mb: 3 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>Trending Words (Bar Chart)</Typography>
              {trendingWords.length > 0 ? (
                <ResponsiveContainer width="100%" height={250}>
                  <BarChart data={trendingWords} margin={{ top: 20, right: 30, left: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                    <XAxis dataKey="word" stroke="#aaa" />
                    <YAxis stroke="#aaa" />
                    <Tooltip />
                    <Bar dataKey="frequency" fill="#42a5f5" name="Frequency" />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <Typography color="#aaa">No trending words available.</Typography>
              )}
            </CardContent>
          </Card>
          <Paper sx={{ p: 3, bgcolor: '#23293a', minHeight: 300 }}>
            <Typography variant="h6" color="#fff" mb={2}>Trending Words (Word Cloud)</Typography>
            <Typography color="#aaa">Word cloud is temporarily unavailable due to compatibility issues. All other analytics are live.</Typography>
          </Paper>
        </Grid>
        <Grid item xs={12} md={4}>
          <Card sx={{ bgcolor: '#23293a', color: '#fff', borderRadius: 3, boxShadow: 3, height: '100%' }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>Recent Activity</Typography>
              {activityFeed.length > 0 ? (
                <List>
                  {activityFeed.map((post, i) => (
                    <ListItem key={i} alignItems="flex-start">
                      <ListItemAvatar>
                        <Avatar sx={{ bgcolor: '#42a5f5' }}><ForumIcon /></Avatar>
                      </ListItemAvatar>
                      <ListItemText
                        primary={post.title || post.text || 'Untitled Post'}
                        secondary={
                          <>
                            <Typography component="span" variant="body2" color="#aaa">
                              {post.author || 'Unknown'}
                            </Typography>
                            {post.created_at && (
                              <Typography component="span" variant="caption" color="#888" sx={{ ml: 1 }}>
                                {new Date(post.created_at).toLocaleString()}
                              </Typography>
                            )}
                          </>
                        }
                      />
                    </ListItem>
                  ))}
                </List>
              ) : (
                <Typography color="#aaa">No recent activity available.</Typography>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
} 