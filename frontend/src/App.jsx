import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Activity, Play, CheckCircle, TrendingUp, AlertTriangle, ArrowRight, DollarSign, Percent, RefreshCw } from 'lucide-react';
import './index.css';

const API_BASE = 'http://localhost:5000/api';

function App() {
  const [config, setConfig] = useState({
    ticker: 'AAPL',
    episodes: 50,
    initial_balance: 10000
  });
  
  const [jobId, setJobId] = useState(null);
  const [status, setStatus] = useState('idle'); // idle, queued, downloading data, training, evaluating, stopping, completed, error, cancelled
  const [progress, setProgress] = useState(0);
  const [logs, setLogs] = useState([]);
  const [results, setResults] = useState(null);
  
  const terminalRef = useRef(null);

  // Auto-scroll terminal
  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [logs]);

  // Polling logic
  useEffect(() => {
    let intervalId;
    
    const checkStatus = async () => {
      if (!jobId) return;
      
      try {
        const res = await axios.get(`${API_BASE}/status/${jobId}`);
        const data = res.data;
        
        setStatus(data.status);
        setProgress(data.progress);
        
        if (data.logs && data.logs.length > 0) {
          setLogs(prev => {
            // Merge unique logs simply by checking last few (this is naive but works for demo)
            const newLogs = [...data.logs];
            return Array.from(new Set([...prev, ...newLogs].map(JSON.stringify))).map(JSON.parse);
          });
        }
        
        if (['completed', 'error', 'cancelled'].includes(data.status)) {
          clearInterval(intervalId);
          if (data.status === 'completed') {
            setResults({
              metrics: data.metrics,
              charts: data.charts
            });
          }
        }
      } catch (err) {
        console.error("Error fetching status:", err);
      }
    };

    if (jobId && !['completed', 'error', 'cancelled'].includes(status)) {
      intervalId = setInterval(checkStatus, 1000);
    }
    
    return () => clearInterval(intervalId);
  }, [jobId, status]);

  const handleStartTraining = async () => {
    try {
      setStatus('queued');
      setProgress(0);
      setLogs([]);
      setResults(null);
      
      const res = await axios.post(`${API_BASE}/train`, {
        ticker: config.ticker,
        episodes: parseInt(config.episodes),
        initial_balance: parseFloat(config.initial_balance),
        lr: parseFloat(config.lr || 0.0001),
        window_size: parseInt(config.window_size || 10)
      });
      
      setJobId(res.data.job_id);
    } catch (err) {
      setStatus('error');
      setLogs(prev => [...prev, `Failed to start: ${err.message}`]);
    }
  };

  const handleConfigChange = (e) => {
    setConfig({
      ...config,
      [e.target.name]: e.target.value
    });
  };

  const handleStopTraining = async () => {
    if (!jobId) return;
    try {
      await axios.post(`${API_BASE}/stop/${jobId}`);
      setStatus('stopping');
      setLogs(prev => [...prev, 'Stop requested...']);
    } catch (err) {
      setLogs(prev => [...prev, `Failed to stop: ${err.message}`]);
    }
  };

  const getStatusBadge = () => {
    switch(status) {
      case 'idle': return null;
      case 'completed': return <span className="status-badge" style={{color: 'var(--accent-primary)', background: 'rgba(0,212,170,0.1)', borderColor: 'rgba(0,212,170,0.2)'}}>Completed</span>;
      case 'error': return <span className="status-badge" style={{color: 'var(--accent-danger)', background: 'rgba(248,81,73,0.1)', borderColor: 'rgba(248,81,73,0.2)'}}>Error</span>;
      case 'cancelled': return <span className="status-badge" style={{color: 'var(--accent-warning)', background: 'rgba(255,166,87,0.12)', borderColor: 'rgba(255,166,87,0.25)'}}>Cancelled</span>;
      case 'stopping': return <span className="status-badge" style={{color: '#79c0ff', background: 'rgba(121,192,255,0.12)', borderColor: 'rgba(121,192,255,0.25)'}}>Stopping...</span>;
      default: return <span className="status-badge" style={{display: 'flex', alignItems: 'center', gap: '0.5rem'}}><span className="loader" style={{width: '12px', height: '12px', borderWidth: '2px'}}></span> {status}</span>;
    }
  };

  const isRunning = !['idle', 'completed', 'error', 'cancelled'].includes(status);

  return (
    <div className="app-container">
      <header className="app-header">
        <h1 className="title">Deep Q-Network Trading Bot</h1>
        <p className="subtitle">AI-powered algorithmic trading simulator using PyTorch & DQN</p>
      </header>

      {/* Configuration Panel */}
      <div className="config-panel">
        <div className="config-grid">
          <div className="input-group">
            <label>Asset Ticker</label>
            <select 
              name="ticker" 
              value={config.ticker} 
              onChange={handleConfigChange} 
              disabled={status !== 'idle' && status !== 'completed' && status !== 'error'}
              className="custom-select"
            >
              <option value="AAPL">Apple (AAPL)</option>
              <option value="MSFT">Microsoft (MSFT)</option>
              <option value="TSLA">Tesla (TSLA)</option>
              <option value="GOOGL">Google (GOOGL)</option>
              <option value="AMZN">Amazon (AMZN)</option>
              <option value="BTC-USD">Bitcoin (BTC-USD)</option>
              <option value="ETH-USD">Ethereum (ETH-USD)</option>
              <option value="SPY">S&P 500 ETF (SPY)</option>
            </select>
          </div>
          <div className="input-group">
            <label>Training Episodes</label>
            <input 
              type="number" 
              name="episodes" 
              value={config.episodes} 
              onChange={handleConfigChange}
              disabled={status !== 'idle' && status !== 'completed' && status !== 'error'}
            />
          </div>
          <div className="input-group">
            <label>Initial Capital ($)</label>
            <input 
              type="number" 
              name="initial_balance" 
              value={config.initial_balance} 
              onChange={handleConfigChange}
              disabled={status !== 'idle' && status !== 'completed' && status !== 'error'}
            />
          </div>
          <div className="input-group">
            <label>Learning Rate</label>
            <select 
              name="lr" 
              value={config.lr || 0.0001} 
              onChange={handleConfigChange} 
              disabled={status !== 'idle' && status !== 'completed' && status !== 'error'}
              className="custom-select"
            >
              <option value={0.001}>High (0.001)</option>
              <option value={0.0001}>Standard (0.0001)</option>
              <option value={0.00001}>Fine-Tune (0.00001)</option>
            </select>
          </div>
          <div className="input-group">
            <label>State Window Size (Days)</label>
            <input 
              type="number" 
              name="window_size" 
              value={config.window_size || 10} 
              onChange={handleConfigChange}
              disabled={status !== 'idle' && status !== 'completed' && status !== 'error'}
            />
          </div>
        </div>
        
        <button 
          className="btn-primary"
          onClick={handleStartTraining}
          disabled={status !== 'idle' && status !== 'completed' && status !== 'error'}
        >
          {status !== 'idle' && status !== 'completed' && status !== 'error' ? (
            <>Initializing <span className="loader" style={{borderColor: '#000', borderTopColor: 'transparent'}}></span></>
          ) : (
            <>Launch Training Agent <Play size={20} /></>
          )}
        </button>
        {isRunning && (
          <button
            className="btn-primary"
            onClick={handleStopTraining}
            disabled={status === 'stopping'}
            style={{marginLeft: '0.75rem', background: 'var(--accent-danger)', opacity: status === 'stopping' ? 0.7 : 1}}
          >
            Stop Training <AlertTriangle size={18} />
          </button>
        )}
      </div>

      {/* Status & Terminal Panel */}
      {jobId && (
        <div className="status-panel">
          <div className="status-header">
            <h2 className="status-title"><Activity size={24} color="var(--accent-primary)" /> Execution Console</h2>
            {getStatusBadge()}
          </div>
          
          <div className="progress-container">
            <div className="progress-bar" style={{ width: `${progress}%` }}></div>
          </div>
          
          <div className="terminal" ref={terminalRef}>
            {logs.map((log, i) => (
              <div key={i} className="terminal-line">{log}</div>
            ))}
            {status !== 'completed' && status !== 'error' && status !== 'cancelled' && (
              <div className="terminal-line" style={{animation: 'pulse 1.5s infinite'}}>_</div>
            )}
          </div>
        </div>
      )}

      {/* Results Dashboard */}
      {results && results.metrics && (
        <div className="results-dashboard">
          <div className="metrics-grid">
            <div className="metric-card">
              <DollarSign size={24} color="var(--accent-primary)" style={{marginBottom: '1rem'}} />
              <div className="metric-label">Final Portfolio Value</div>
              <div className="metric-value positive">
                ${results.metrics.dqn['Final Portfolio ($)'].toLocaleString()}
              </div>
            </div>
            
            <div className="metric-card">
              <Percent size={24} color={results.metrics.dqn['Total Return (%)'] >= 0 ? "var(--accent-primary)" : "var(--accent-danger)"} style={{marginBottom: '1rem'}} />
              <div className="metric-label">DQN Total Return</div>
              <div className={`metric-value ${results.metrics.dqn['Total Return (%)'] >= 0 ? 'positive' : 'negative'}`}>
                {results.metrics.dqn['Total Return (%)']}%
              </div>
            </div>

            <div className="metric-card">
              <TrendingUp size={24} color="var(--accent-secondary)" style={{marginBottom: '1rem'}} />
              <div className="metric-label">Buy & Hold Return</div>
              <div className={`metric-value ${results.metrics.bh_ret >= 0 ? 'positive' : 'negative'}`}>
                {results.metrics.bh_ret.toFixed(2)}%
              </div>
            </div>

            <div className="metric-card">
              <CheckCircle size={24} color={results.metrics.alpha > 0 ? "var(--accent-primary)" : "var(--accent-warning)"} style={{marginBottom: '1rem'}} />
              <div className="metric-label">Alpha Generated</div>
              <div className={`metric-value ${results.metrics.alpha > 0 ? 'positive' : 'negative'}`}>
                {results.metrics.alpha > 0 ? '+' : ''}{results.metrics.alpha.toFixed(2)}%
              </div>
            </div>
          </div>

          {results.charts && results.charts.performance && (
            <div className="chart-container">
              <div className="chart-header">
                <h3 className="chart-title">Trading Performance & Action Strategy</h3>
              </div>
              <img src={results.charts.performance} alt="Trading Performance" className="chart-image" />
            </div>
          )}

          {results.charts && results.charts.training && (
            <div className="chart-container">
              <div className="chart-header">
                <h3 className="chart-title">DQN Training Dynamics</h3>
              </div>
              <img src={results.charts.training} alt="Training Curves" className="chart-image" />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
