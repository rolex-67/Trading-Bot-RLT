import os

from trading_bot import (
    DataProcessor,
    TradingEnvironment,
    DQNAgent,
    buy_and_hold,
    plot_training,
    plot_trading_performance,
)


class TrainingCancelled(Exception):
    pass


def _check_cancel(cancel_event, state):
    if cancel_event.is_set():
        state['status'] = 'cancelled'
        state['logs'].append("Training cancelled by user.")
        raise TrainingCancelled()


def run_training_task(state, cancel_event, config):
    try:
        _check_cancel(cancel_event, state)

        state['status'] = 'downloading data'
        state['logs'].append("Downloading data...")

        processor = DataProcessor(config)
        raw_df = processor.download()
        _check_cancel(cancel_event, state)
        state['logs'].append(f"Data downloaded: {len(raw_df)} days.")

        feat_df = processor.add_features(raw_df.copy())
        _check_cancel(cancel_event, state)
        train_df, test_df = processor.split(feat_df)
        state['logs'].append("Features added. Split into train/test.")

        norm_cols = processor.norm_cols
        state_size = len(norm_cols) + 3
        action_size = 3

        agent = DQNAgent(state_size, action_size, config)
        _check_cancel(cancel_event, state)

        state['status'] = 'training'
        state['logs'].append(f"Starting training for {config['episodes']} episodes.")

        env = TradingEnvironment(train_df, norm_cols, config)

        episode_returns = []
        episode_values = []
        losses_per_ep = []
        best_value = -float('inf')

        for ep in range(config['episodes']):
            _check_cancel(cancel_event, state)

            state_val = env.reset()
            total_reward = 0.0
            ep_losses = []

            while True:
                _check_cancel(cancel_event, state)

                action = agent.select_action(state_val)
                next_state, rew, done = env.step(action)
                agent.store(state_val, action, rew, next_state, done)
                loss = agent.learn()
                if loss > 0:
                    ep_losses.append(loss)

                state_val = next_state
                total_reward += rew

                if done:
                    break

            _check_cancel(cancel_event, state)

            agent.update_epsilon()
            agent.epsilons.append(agent.epsilon)

            if (ep + 1) % config['target_update'] == 0:
                agent.update_target()

            agent.scheduler.step()
            metrics = env.get_metrics()
            final_v = metrics.get("Final Portfolio ($)", config['initial_balance'])

            episode_returns.append(total_reward)
            episode_values.append(final_v)
            losses_per_ep.append(sum(ep_losses) / len(ep_losses) if ep_losses else 0)

            if final_v > best_value:
                best_value = final_v
                agent.save("dqn_best.pt")

            state['progress'] = (ep + 1) / config['episodes'] * 100
            state['logs'].append(
                f"Episode {ep + 1}/{config['episodes']} | Value: ${final_v:,.2f} | ε: {agent.epsilon:.2f}"
            )

        _check_cancel(cancel_event, state)

        state['status'] = 'evaluating'
        state['logs'].append("Evaluating on Test Data...")

        training_chart = plot_training(episode_returns, episode_values, losses_per_ep, agent, config)
        _check_cancel(cancel_event, state)
        state['charts']['training'] = training_chart

        if os.path.exists("dqn_best.pt"):
            agent.load("dqn_best.pt")
        agent.epsilon = 0.0

        test_env = TradingEnvironment(test_df, norm_cols, config)
        state_val = test_env.reset()
        while True:
            _check_cancel(cancel_event, state)
            action = agent.select_action(state_val)
            state_val, _, done = test_env.step(action)
            if done:
                break

        test_metrics = test_env.get_metrics()
        bh_val, bh_ret = buy_and_hold(test_df, config)
        _check_cancel(cancel_event, state)

        perf_chart = plot_trading_performance(test_env, test_df, config)
        state['charts']['performance'] = perf_chart

        state['metrics']['dqn'] = dict(test_metrics)
        state['metrics']['bh_val'] = bh_val
        state['metrics']['bh_ret'] = bh_ret
        state['metrics']['alpha'] = test_metrics.get('Total Return (%)', 0) - bh_ret

        state['status'] = 'completed'
        state['logs'].append("Training and Evaluation complete.")

    except TrainingCancelled:
        pass
    except Exception as e:
        if not cancel_event.is_set():
            state['status'] = 'error'
            state['logs'].append(f"Error: {str(e)}")
