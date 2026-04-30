from flask import Flask, jsonify, request
from flask_cors import CORS
import threading
import uuid
import time
import os
import ctypes

# Import everything from our trading_bot module
from trading_bot import (
    get_base_config, DataProcessor, TradingEnvironment,
    DQNAgent, buy_and_hold, plot_training, plot_trading_performance
)

app = Flask(__name__)
CORS(app)

# Global dict to hold training jobs
# jobs[job_id] = { status, progress, metrics, charts, logs }
jobs = {}

def is_cancelled(job_id):
    return jobs.get(job_id, {}).get('cancel_requested', False)

def _terminate_thread(thread_obj):
    """
    Force-stop a Python thread by asynchronously raising SystemExit in it.
    Use only for user-initiated cancellation of long-running training jobs.
    """
    if not thread_obj or not thread_obj.is_alive():
        return False

    thread_id = thread_obj.ident
    if thread_id is None:
        return False

    result = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_ulong(thread_id),
        ctypes.py_object(SystemExit)
    )

    if result == 0:
        return False
    if result > 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(thread_id), None)
        return False
    return True

def run_training_task(job_id, config):
    try:
        if is_cancelled(job_id):
            jobs[job_id]['status'] = 'cancelled'
            jobs[job_id]['logs'].append("Training cancelled before start.")
            return

        jobs[job_id]['status'] = 'downloading data'
        jobs[job_id]['logs'].append("Downloading data...")
        
        processor = DataProcessor(config)
        raw_df = processor.download()
        jobs[job_id]['logs'].append(f"Data downloaded: {len(raw_df)} days.")
        
        feat_df = processor.add_features(raw_df.copy())
        train_df, test_df = processor.split(feat_df)
        jobs[job_id]['logs'].append("Features added. Split into train/test.")
        
        NORM_COLS = processor.norm_cols
        STATE_SIZE = len(NORM_COLS) + 3
        ACTION_SIZE = 3
        
        agent = DQNAgent(STATE_SIZE, ACTION_SIZE, config)
        
        jobs[job_id]['status'] = 'training'
        jobs[job_id]['logs'].append(f"Starting training for {config['episodes']} episodes.")
        
        env = TradingEnvironment(train_df, NORM_COLS, config)
        
        episode_returns = []
        episode_values = []
        losses_per_ep = []
        best_value = -float('inf')
        
        for ep in range(config['episodes']):
            if is_cancelled(job_id):
                jobs[job_id]['status'] = 'cancelled'
                jobs[job_id]['logs'].append("Training cancelled by user.")
                return

            state = env.reset()
            total_reward = 0.0
            ep_losses = []
            
            while True:
                if is_cancelled(job_id):
                    jobs[job_id]['status'] = 'cancelled'
                    jobs[job_id]['logs'].append("Training cancelled by user.")
                    return

                action = agent.select_action(state)
                next_state, rew, done = env.step(action)
                agent.store(state, action, rew, next_state, done)
                loss = agent.learn()
                if loss > 0:
                    ep_losses.append(loss)
                
                state = next_state
                total_reward += rew
                
                if done:
                    break
                    
            agent.update_epsilon()
            agent.epsilons.append(agent.epsilon)
            
            if (ep + 1) % config['target_update'] == 0:
                agent.update_target()
                
            agent.scheduler.step()
            metrics = env.get_metrics()
            final_v = metrics.get("Final Portfolio ($)", config['initial_balance'])
            
            episode_returns.append(total_reward)
            episode_values.append(final_v)
            losses_per_ep.append(sum(ep_losses)/len(ep_losses) if ep_losses else 0)
            
            if final_v > best_value:
                best_value = final_v
                agent.save("dqn_best.pt")
                
            jobs[job_id]['progress'] = (ep + 1) / config['episodes'] * 100
            jobs[job_id]['logs'].append(f"Episode {ep+1}/{config['episodes']} | Value: ${final_v:,.2f} | ε: {agent.epsilon:.2f}")

        # Generate Training Plot
        if is_cancelled(job_id):
            jobs[job_id]['status'] = 'cancelled'
            jobs[job_id]['logs'].append("Training cancelled by user.")
            return

        jobs[job_id]['status'] = 'evaluating'
        jobs[job_id]['logs'].append("Evaluating on Test Data...")
        
        training_chart = plot_training(episode_returns, episode_values, losses_per_ep, agent, config)
        jobs[job_id]['charts']['training'] = training_chart
        
        # Evaluate
        if os.path.exists("dqn_best.pt"):
            agent.load("dqn_best.pt")
        agent.epsilon = 0.0
        
        test_env = TradingEnvironment(test_df, NORM_COLS, config)
        state = test_env.reset()
        while True:
            if is_cancelled(job_id):
                jobs[job_id]['status'] = 'cancelled'
                jobs[job_id]['logs'].append("Training cancelled by user.")
                return
            action = agent.select_action(state)
            state, _, done = test_env.step(action)
            if done: break
            
        test_metrics = test_env.get_metrics()
        bh_val, bh_ret = buy_and_hold(test_df, config)
        
        perf_chart = plot_trading_performance(test_env, test_df, config)
        jobs[job_id]['charts']['performance'] = perf_chart
        
        jobs[job_id]['metrics'] = {
            'dqn': test_metrics,
            'bh_val': bh_val,
            'bh_ret': bh_ret,
            'alpha': test_metrics.get('Total Return (%)', 0) - bh_ret
        }
        
        jobs[job_id]['status'] = 'completed'
        jobs[job_id]['logs'].append("Training and Evaluation complete.")
        
    except SystemExit:
        jobs[job_id]['status'] = 'cancelled'
        jobs[job_id]['logs'].append("Training force-stopped by user.")
        return
    except Exception as e:
        jobs[job_id]['status'] = 'error'
        jobs[job_id]['logs'].append(f"Error: {str(e)}")

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'service': 'DQN Trading Bot API',
        'status': 'online',
        'version': '1.0.0',
        'description': 'Backend service for training and evaluating a Deep Q-Network trading agent.',
        'endpoints': {
            'health': 'GET /',
            'start_training': 'POST /api/train',
            'job_status': 'GET /api/status/<job_id>',
            'stop_training': 'POST /api/stop/<job_id>'
        },
        'notes': [
            'Run frontend separately from /frontend using Vite.',
            'This endpoint is informational and intended for service verification.'
        ]
    })

@app.route('/api/train', methods=['POST'])
def start_training():
    req = request.json or {}
    config = get_base_config()
    # Override defaults with user request
    for k in config.keys():
        if k in req:
            config[k] = req[k]
            
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        'status': 'queued',
        'progress': 0,
        'logs': [],
        'charts': {},
        'metrics': {},
        'config': config,
        'cancel_requested': False
    }
    
    t = threading.Thread(target=run_training_task, args=(job_id, config))
    t.daemon = True
    jobs[job_id]['thread'] = t
    t.start()
    
    return jsonify({'job_id': job_id})

@app.route('/api/stop/<job_id>', methods=['POST'])
def stop_training(job_id):
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404

    job = jobs[job_id]
    if job['status'] in ('completed', 'error', 'cancelled'):
        return jsonify({'message': f"Job already {job['status']}", 'status': job['status']})

    job['cancel_requested'] = True
    job['logs'].append("Stop requested by user.")
    job['status'] = 'stopping'

    thread_obj = job.get('thread')
    force_stopped = _terminate_thread(thread_obj)
    if force_stopped:
        job['status'] = 'cancelled'
        job['logs'].append("Training force-stopped successfully.")
        return jsonify({'message': 'Training force-stopped', 'status': job['status']})

    return jsonify({'message': 'Stop requested; awaiting graceful shutdown', 'status': job['status']})

@app.route('/api/status/<job_id>', methods=['GET'])
def get_status(job_id):
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
        
    job = jobs[job_id]
    return jsonify({
        'status': job['status'],
        'progress': job['progress'],
        'logs': job['logs'][-10:], # Return last 10 logs
        'metrics': job['metrics'],
        'charts': job['charts'] if job['status'] == 'completed' else {}
    })

if __name__ == '__main__':
    app.run(port=5000, debug=False, use_reloader=False)
