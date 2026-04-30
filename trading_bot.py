import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import random
import warnings
import os
import io
import base64
from collections import deque, namedtuple
import yfinance as yf

warnings.filterwarnings("ignore")

# ─── DEVICE ───
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_base_config():
    return {
        "ticker":            "AAPL",       
        "start_date":        "2018-01-01",
        "end_date":          "2023-12-31",
        "initial_balance":   10_000.0,     
        "transaction_cost":  0.001,        
        "episodes":          50,          # Reduced for web demo
        "batch_size":        64,
        "gamma":             0.99,         
        "lr":                1e-4,         
        "epsilon_start":     1.0,
        "epsilon_end":       0.01,
        "epsilon_decay":     0.995,
        "memory_size":       10_000,
        "target_update":     10,           
        "hidden_units":      [256, 128, 64],
        "train_split":       0.80,         
        "window_size":       10,           
    }

class DataProcessor:
    def __init__(self, cfg):
        self.cfg = cfg

    def download(self):
        df = yf.download(
            self.cfg['ticker'],
            start=self.cfg['start_date'],
            end=self.cfg['end_date'],
            progress=False
        )
        if df.empty:
            raise ValueError(f"No data found for {self.cfg['ticker']}")
        df.dropna(inplace=True)
        return df

    def add_features(self, df):
        # Flatten columns if MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        close = df['Close'].squeeze()
        high  = df['High'].squeeze()
        low   = df['Low'].squeeze()
        vol   = df['Volume'].squeeze()

        df['MA5']  = close.rolling(5).mean()
        df['MA10'] = close.rolling(10).mean()
        df['MA20'] = close.rolling(20).mean()
        df['MA50'] = close.rolling(50).mean()

        df['EMA12'] = close.ewm(span=12, adjust=False).mean()
        df['EMA26'] = close.ewm(span=26, adjust=False).mean()

        df['MACD']        = df['EMA12'] - df['EMA26']
        df['MACD_Signal']  = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist']    = df['MACD'] - df['MACD_Signal']

        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / (loss + 1e-8)
        df['RSI'] = 100 - 100 / (1 + rs)

        mb = close.rolling(20).mean()
        sb = close.rolling(20).std()
        df['BB_Upper'] = mb + 2 * sb
        df['BB_Lower'] = mb - 2 * sb
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / (mb + 1e-8)
        df['BB_Pct']   = (close - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'] + 1e-8)

        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low  - close.shift()).abs()
        df['ATR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()

        obv = [0]
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.append(obv[-1] + vol.iloc[i])
            elif close.iloc[i] < close.iloc[i-1]:
                obv.append(obv[-1] - vol.iloc[i])
            else:
                obv.append(obv[-1])
        df['OBV'] = obv

        df['Return_1d'] = close.pct_change(1)
        df['Return_5d'] = close.pct_change(5)

        df['Volatility'] = df['Return_1d'].rolling(10).std()

        df.dropna(inplace=True)

        feature_cols = [
            'Close','Open','High','Low','Volume',
            'MA5','MA10','MA20','MA50',
            'EMA12','EMA26','MACD','MACD_Signal','MACD_Hist',
            'RSI','BB_Width','BB_Pct','ATR','OBV',
            'Return_1d','Return_5d','Volatility'
        ]
        self.feature_cols = feature_cols
        for col in feature_cols:
            mn = df[col].min()
            mx = df[col].max()
            df[col + '_norm'] = (df[col] - mn) / (mx - mn + 1e-8)

        self.norm_cols = [c + '_norm' for c in feature_cols]
        return df

    def split(self, df):
        n = int(len(df) * self.cfg['train_split'])
        train = df.iloc[:n].reset_index(drop=True)
        test  = df.iloc[n:].reset_index(drop=True)
        return train, test


class TradingEnvironment:
    def __init__(self, df, norm_cols, cfg):
        self.df         = df
        self.norm_cols  = norm_cols
        self.cfg        = cfg
        self.n_steps    = len(df)
        self.reset()

    def reset(self):
        self.step_idx    = 0
        self.balance     = self.cfg['initial_balance']
        self.holdings    = 0
        self.total_value = self.balance
        self.history     = []
        self.trades      = []
        return self._get_state()

    def _get_state(self):
        row = self.df.iloc[self.step_idx]
        market_features = row[self.norm_cols].values.astype(np.float32)

        price = float(self.df['Close'].iloc[self.step_idx])
        stock_val = self.holdings * price
        total     = self.balance + stock_val + 1e-8
        portfolio_state = np.array([
            self.balance / self.cfg['initial_balance'],
            stock_val    / self.cfg['initial_balance'],
            stock_val    / total
        ], dtype=np.float32)

        return np.concatenate([market_features, portfolio_state])

    def step(self, action):
        price = float(self.df['Close'].iloc[self.step_idx])
        cost  = self.cfg['transaction_cost']
        prev_value = self.balance + self.holdings * price

        if action == 1:   # BUY
            if self.balance > price:
                shares_to_buy     = int(self.balance * (1 - cost) / price)
                if shares_to_buy > 0:
                    self.holdings  += shares_to_buy
                    self.balance   -= shares_to_buy * price * (1 + cost)
                    self.trades.append(('BUY', self.step_idx, price, shares_to_buy))

        elif action == 2: # SELL
            if self.holdings > 0:
                proceeds          = self.holdings * price * (1 - cost)
                self.trades.append(('SELL', self.step_idx, price, self.holdings))
                self.balance     += proceeds
                self.holdings     = 0

        self.step_idx += 1
        done = (self.step_idx >= self.n_steps - 1)

        next_price = float(self.df['Close'].iloc[self.step_idx])
        curr_value = self.balance + self.holdings * next_price
        self.total_value = curr_value

        reward = np.log(curr_value / (prev_value + 1e-8))

        self.history.append({
            'step':      self.step_idx,
            'price':     next_price,
            'balance':   self.balance,
            'holdings':  self.holdings,
            'value':     curr_value,
            'action':    action,
        })

        next_state = self._get_state()
        return next_state, reward, done

    def get_metrics(self):
        if not self.history:
            return {}
        values = [h['value'] for h in self.history]
        init   = self.cfg['initial_balance']
        total_return = (values[-1] - init) / init * 100

        daily_rets = pd.Series(values).pct_change().dropna()
        sharpe = (daily_rets.mean() / (daily_rets.std() + 1e-8)) * np.sqrt(252)

        peak = pd.Series(values).cummax()
        dd   = (pd.Series(values) - peak) / (peak + 1e-8)
        max_dd = dd.min() * 100

        return {
            "Final Portfolio ($)": round(values[-1], 2),
            "Total Return (%)":    round(total_return, 2),
            "Sharpe Ratio":        round(sharpe, 4),
            "Max Drawdown (%)":    round(max_dd, 2),
            "Num Trades":          len(self.trades),
        }

class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_units):
        super(DQN, self).__init__()
        layers = []
        in_dim = state_size
        for h in hidden_units:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(0.1)]
            in_dim = h
        self.feature_net = nn.Sequential(*layers)

        self.value_stream     = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, action_size)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        features   = self.feature_net(x)
        value      = self.value_stream(features)
        advantage  = self.advantage_stream(features)
        q_values   = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQNAgent:
    def __init__(self, state_size, action_size, cfg):
        self.state_size  = state_size
        self.action_size = action_size
        self.cfg         = cfg

        self.epsilon     = cfg['epsilon_start']
        self.eps_end     = cfg['epsilon_end']
        self.eps_decay   = cfg['epsilon_decay']
        self.gamma       = cfg['gamma']
        self.batch_size  = cfg['batch_size']

        self.policy_net = DQN(state_size, action_size, cfg['hidden_units']).to(DEVICE)
        self.target_net = DQN(state_size, action_size, cfg['hidden_units']).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer  = optim.Adam(self.policy_net.parameters(), lr=cfg['lr'])
        self.scheduler  = optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.5)
        self.memory     = ReplayMemory(cfg['memory_size'])
        self.loss_fn    = nn.SmoothL1Loss()   # Huber loss

        self.epsilons     = []

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            q_vals = self.policy_net(state_t)
        return q_vals.argmax().item()

    def store(self, state, action, reward, next_state, done):
        self.memory.push(
            torch.FloatTensor(state),
            torch.LongTensor([action]),
            torch.FloatTensor([reward]),
            torch.FloatTensor(next_state),
            torch.BoolTensor([done])
        )

    def learn(self):
        if len(self.memory) < self.batch_size:
            return 0.0

        transitions = self.memory.sample(self.batch_size)
        batch       = Transition(*zip(*transitions))

        states      = torch.stack(batch.state).to(DEVICE)
        actions     = torch.cat(batch.action).to(DEVICE)
        rewards     = torch.cat(batch.reward).to(DEVICE)
        next_states = torch.stack(batch.next_state).to(DEVICE)
        dones       = torch.cat(batch.done).to(DEVICE)

        q_vals    = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()

        with torch.no_grad():
            next_actions    = self.policy_net(next_states).argmax(1)
            next_q          = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q        = rewards + self.gamma * next_q * (~dones)

        loss = self.loss_fn(q_vals, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()

    def update_epsilon(self):
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path="dqn_best.pt"):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer':  self.optimizer.state_dict(),
            'epsilon':    self.epsilon,
        }, path)

    def load(self, path="dqn_best.pt"):
        ckpt = torch.load(path, map_location=DEVICE, weights_only=True)
        self.policy_net.load_state_dict(ckpt['policy_net'])
        self.target_net.load_state_dict(ckpt['target_net'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.epsilon = ckpt['epsilon']

def buy_and_hold(df, cfg):
    price_start = float(df['Close'].iloc[0])
    price_end   = float(df['Close'].iloc[-1])
    shares      = int(cfg['initial_balance'] / price_start)
    final       = shares * price_end + (cfg['initial_balance'] - shares * price_start)
    ret         = (final - cfg['initial_balance']) / cfg['initial_balance'] * 100
    return final, ret

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#0d1117')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{img_str}"

def plot_training(ep_rewards, ep_values, ep_losses, agent, config):
    fig = plt.figure(figsize=(15, 10))
    fig.patch.set_facecolor('#0d1117')
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25)

    style = {'color': '#00d4aa', 'linewidth': 1.5}
    ax_style = {'facecolor': '#161b22', 'grid_color': '#30363d'}

    def setup_ax(ax, title, xlabel, ylabel):
        ax.set_facecolor(ax_style['facecolor'])
        ax.set_title(title, color='#e6edf3', fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel(xlabel, color='#8b949e', fontsize=9)
        ax.set_ylabel(ylabel, color='#8b949e', fontsize=9)
        ax.tick_params(colors='#8b949e')
        ax.spines[:].set_color('#30363d')
        ax.grid(color='#30363d', linestyle='--', alpha=0.5)

    ax1 = fig.add_subplot(gs[0, 0])
    setup_ax(ax1, "Episode Total Reward", "Episode", "Reward")
    smooth = pd.Series(ep_rewards).rolling(max(1, len(ep_rewards)//10)).mean()
    ax1.plot(ep_rewards, alpha=0.3, color='#58a6ff', linewidth=0.8)
    ax1.plot(smooth, color='#00d4aa', linewidth=2, label='MA')
    ax1.axhline(0, color='#f85149', linestyle='--', alpha=0.6)
    ax1.legend(facecolor='#161b22', labelcolor='#e6edf3', fontsize=8)

    ax2 = fig.add_subplot(gs[0, 1])
    setup_ax(ax2, "Portfolio Value per Episode ($)", "Episode", "Value ($)")
    smooth_v = pd.Series(ep_values).rolling(max(1, len(ep_values)//10)).mean()
    ax2.plot(ep_values, alpha=0.3, color='#58a6ff', linewidth=0.8)
    ax2.plot(smooth_v, color='#ffa657', linewidth=2, label='MA')
    ax2.axhline(config['initial_balance'], color='#f85149', linestyle='--',
                alpha=0.7, label='Initial Capital')
    ax2.legend(facecolor='#161b22', labelcolor='#e6edf3', fontsize=8)

    ax3 = fig.add_subplot(gs[1, 0])
    setup_ax(ax3, "Training Loss", "Episode", "Loss")
    ax3.plot(ep_losses, color='#d2a8ff', linewidth=1.3)

    ax4 = fig.add_subplot(gs[1, 1])
    setup_ax(ax4, "Epsilon Decay", "Episode", "Epsilon")
    ax4.plot(agent.epsilons, color='#79c0ff', linewidth=1.8)
    ax4.fill_between(range(len(agent.epsilons)), agent.epsilons, alpha=0.15, color='#79c0ff')
    
    return fig_to_base64(fig)

def plot_trading_performance(env, df, cfg):
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), facecolor='#0d1117')
    fig.patch.set_facecolor('#0d1117')

    history = env.history
    if not history:
        return ""
    steps   = [h['step'] for h in history]
    prices  = [h['price'] for h in history]
    values  = [h['value'] for h in history]

    start_price = float(df['Close'].iloc[0])
    shares_bh   = cfg['initial_balance'] / start_price
    bh_curve    = [shares_bh * p for p in prices]

    ax1 = axes[0]
    ax1.set_facecolor('#161b22')
    ax1.plot(steps, values,   color='#00d4aa', linewidth=2,   label='DQN Agent')
    ax1.plot(steps, bh_curve, color='#ffa657', linewidth=1.5, linestyle='--', label='Buy-and-Hold')
    ax1.axhline(cfg['initial_balance'], color='#f85149', linestyle=':', alpha=0.7, label='Initial Capital')
    ax1.fill_between(steps, values, bh_curve, where=[v > b for v,b in zip(values,bh_curve)],
                     alpha=0.12, color='#00d4aa', label='DQN Outperforms')
    ax1.fill_between(steps, values, bh_curve, where=[v < b for v,b in zip(values,bh_curve)],
                     alpha=0.12, color='#f85149')
    ax1.set_title(f"Portfolio Value — DQN vs Buy-and-Hold ({cfg['ticker']})", color='#e6edf3', fontsize=13, fontweight='bold')
    ax1.set_ylabel("Portfolio Value ($)", color='#8b949e')
    ax1.tick_params(colors='#8b949e')
    ax1.spines[:].set_color('#30363d')
    ax1.grid(color='#30363d', linestyle='--', alpha=0.5)
    ax1.legend(facecolor='#161b22', labelcolor='#e6edf3', fontsize=9)

    ax2 = axes[1]
    ax2.set_facecolor('#161b22')
    ax2.plot(steps, prices, color='#58a6ff', linewidth=1.5, label='Close Price', alpha=0.9)
    buy_x  = [h['step'] for h in history if h['action'] == 1]
    buy_y  = [h['price'] for h in history if h['action'] == 1]
    sell_x = [h['step'] for h in history if h['action'] == 2]
    sell_y = [h['price'] for h in history if h['action'] == 2]
    ax2.scatter(buy_x,  buy_y,  marker='^', color='#3fb950', s=60, zorder=5, label='Buy')
    ax2.scatter(sell_x, sell_y, marker='v', color='#f85149', s=60, zorder=5, label='Sell')
    ax2.set_title("Stock Price with Trade Signals", color='#e6edf3', fontsize=13, fontweight='bold')
    ax2.set_ylabel("Price ($)", color='#8b949e')
    ax2.tick_params(colors='#8b949e')
    ax2.spines[:].set_color('#30363d')
    ax2.grid(color='#30363d', linestyle='--', alpha=0.5)
    ax2.legend(facecolor='#161b22', labelcolor='#e6edf3', fontsize=9)

    plt.tight_layout()
    return fig_to_base64(fig)
