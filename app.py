from flask import Flask, jsonify, request
from flask_cors import CORS
import uuid
import os
import threading

from trading_bot import get_base_config
from training_worker import run_training_task

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

jobs = {}


def _create_job_state(config):
    return {
        'status': 'queued',
        'progress': 0.0,
        'logs': [],
        'charts': {},
        'metrics': {},
        'config': config,
    }


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
    for k in config.keys():
        if k in req:
            config[k] = req[k]

    job_id = str(uuid.uuid4())
    state = _create_job_state(config)
    cancel_event = threading.Event()

    thread = threading.Thread(
        target=run_training_task,
        args=(state, cancel_event, config),
        daemon=True,
    )
    jobs[job_id] = {
        'state': state,
        'thread': thread,
        'cancel_event': cancel_event,
    }
    thread.start()

    return jsonify({'job_id': job_id})


@app.route('/api/stop/<job_id>', methods=['POST'])
def stop_training(job_id):
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404

    job = jobs[job_id]
    state = job['state']

    if state['status'] in ('completed', 'error', 'cancelled'):
        return jsonify({'message': f"Job already {state['status']}", 'status': state['status']})

    job['cancel_event'].set()
    state['status'] = 'cancelled'
    state['logs'].append("Training stopped by user.")

    return jsonify({'message': 'Training stopped', 'status': 'cancelled'})


@app.route('/api/status/<job_id>', methods=['GET'])
def get_status(job_id):
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404

    state = jobs[job_id]['state']
    logs = list(state['logs'])
    return jsonify({
        'status': state['status'],
        'progress': state['progress'],
        'logs': logs[-10:],
        'metrics': dict(state['metrics']),
        'charts': dict(state['charts']) if state['status'] == 'completed' else {}
    })


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
