import os
import time
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from environment.custom_env import AntFarmEnv

class RewardLoggingCallback(BaseCallback):
    """Callback to log Episode rewards and lengths"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.rewards = []
        
    def _on_step(self) -> bool:
        if "episode" in self.locals["infos"][0]:
            self.rewards.append(self.locals["infos"][0]["episode"]["r"])
        return True

def train_dqn(learning_rate=1e-3, buffer_size=50000, exploration_fraction=0.1, total_timesteps=20000, experiment_name="default"):
    print(f"--- Training DQN: {experiment_name} ---")
    log_dir = f"./models/dqn/{experiment_name}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create and monitor environment
    env = AntFarmEnv(render_mode=None) # No rendering during fast training
    env = Monitor(env)
    
    # Initialize Model with hyperparameters
    model = DQN(
        "MlpPolicy", 
        env, 
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        exploration_fraction=exploration_fraction,
        verbose=0
    )
    
    # Train Model
    callback = RewardLoggingCallback()
    start_time = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callback)
    
    # Save Model
    model.save(f"{log_dir}/dqn_ant_model")
    env.close()
    
    # Plotting Learning Curve
    plt.figure(figsize=(10, 5))
    plt.plot(callback.rewards)
    plt.title(f"DQN Training - {experiment_name}")
    plt.xlabel('Episodes')
    plt.ylabel('Episode Reward')
    plt.grid(True)
    plt.savefig(f"{log_dir}/learning_curve.png")
    plt.close()
    
    print(f"Finished in {time.time() - start_time:.2f}s. Final average reward (last 10 eps): {sum(callback.rewards[-10:])/min(10, len(callback.rewards)):.2f}")

if __name__ == "__main__":
    # 10 Experiments for DQN Hyperparameter Tuning
    train_dqn(learning_rate=0.0001, buffer_size=10000, exploration_fraction=0.1, experiment_name="dqn_exp_1")
    train_dqn(learning_rate=0.0005, buffer_size=10000, exploration_fraction=0.2, experiment_name="dqn_exp_2")
    train_dqn(learning_rate=0.0010, buffer_size=50000, exploration_fraction=0.1, experiment_name="dqn_exp_3")
    train_dqn(learning_rate=0.0010, buffer_size=50000, exploration_fraction=0.3, experiment_name="dqn_exp_4")
    train_dqn(learning_rate=0.0020, buffer_size=100000, exploration_fraction=0.2, experiment_name="dqn_exp_5")
    train_dqn(learning_rate=0.0020, buffer_size=100000, exploration_fraction=0.5, experiment_name="dqn_exp_6")
    train_dqn(learning_rate=0.0050, buffer_size=10000, exploration_fraction=0.1, experiment_name="dqn_exp_7")
    train_dqn(learning_rate=0.0050, buffer_size=50000, exploration_fraction=0.4, experiment_name="dqn_exp_8")
    train_dqn(learning_rate=0.0100, buffer_size=100000, exploration_fraction=0.3, experiment_name="dqn_exp_9")
    train_dqn(learning_rate=0.0100, buffer_size=200000, exploration_fraction=0.5, experiment_name="dqn_exp_10")

