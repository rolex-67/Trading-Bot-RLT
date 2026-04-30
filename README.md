# Deep Q-Network Trading Bot (RLT)

A full-stack reinforcement learning project for training and evaluating a Deep Q-Network (DQN) agent on financial market data.

This repository includes:
- A **Flask backend** for training orchestration, evaluation, and API exposure.
- A **React + Vite frontend** for experiment configuration, live logs, progress tracking, and chart visualization.
- A **PyTorch DQN engine** for policy learning and portfolio performance comparison against Buy-and-Hold.

## Architecture

- `app.py`: Flask API server and background job controller.
- `trading_bot.py`: RL core logic (data processing, environment, DQN agent, metrics, plotting).
- `frontend/`: React client (Vite) for running and monitoring experiments.

## Key Features

- Asynchronous RL training jobs with real-time polling.
- Mid-run job cancellation support (`/api/stop/<job_id>`).
- Technical indicator engineering (MA, EMA, MACD, RSI, Bollinger, ATR, OBV, volatility).
- DQN vs Buy-and-Hold evaluation with metrics and charts.
- API-first backend suitable for local development and extension.

## Tech Stack

- **Backend:** Python, Flask, Flask-CORS
- **ML/RL:** PyTorch, NumPy, pandas, yfinance
- **Visualization:** matplotlib, seaborn
- **Frontend:** React, Vite, Axios

## Prerequisites

- Python 3.10+ (recommended)
- Node.js 18+ and npm
- PowerShell (commands below are Windows-friendly)

## Local Setup

### 1) Backend setup

```powershell
cd "D:\Projects\Trading Bot Reinforcemet Learning"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install flask flask-cors numpy pandas matplotlib seaborn torch yfinance
python app.py
```

Backend default URL: `http://127.0.0.1:5000`

### 2) Frontend setup

Open a new terminal:

```powershell
cd "D:\Projects\Trading Bot Reinforcemet Learning\frontend"
npm install
npm run dev
```

Frontend default URL: `http://localhost:5173`

## API Endpoints

### Health / Welcome

- `GET /`
- Returns service metadata, API endpoints, and runtime status.

### Start Training

- `POST /api/train`
- Request JSON (example):

```json
{
  "ticker": "AAPL",
  "episodes": 50,
  "initial_balance": 10000,
  "lr": 0.0001,
  "window_size": 10
}
```

- Response:

```json
{
  "job_id": "uuid-string"
}
```

### Check Job Status

- `GET /api/status/<job_id>`
- Returns current status, progress, latest logs, metrics, and charts when complete.

### Stop a Running Job

- `POST /api/stop/<job_id>`
- Requests immediate cancellation of an in-progress job.

## Training Workflow

1. Frontend posts config to `/api/train`.
2. Backend starts a background training worker.
3. Frontend polls `/api/status/<job_id>` every second.
4. Backend streams progress/log updates and returns charts on completion.
5. User can stop the job at any time via `/api/stop/<job_id>`.

## Output Metrics

- Final Portfolio Value
- Total Return (%)
- Sharpe Ratio
- Max Drawdown (%)
- Number of Trades
- Alpha (DQN return - Buy-and-Hold return)

## Notes

- `dqn_best.pt` is generated for the best-performing model checkpoint.
- Data is pulled from Yahoo Finance via `yfinance`.
- For deterministic experiments, seed handling can be added in a future enhancement.

## Troubleshooting

- **`ModuleNotFoundError: flask`**
  - Activate virtual environment and reinstall backend dependencies.
- **Frontend scripts not found**
  - Run npm commands inside the `frontend` directory.
- **PowerShell blocks venv activation**
  - Run:
  ```powershell
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  ```

## License

Add your preferred license (MIT, Apache-2.0, etc.) before open-source distribution.
