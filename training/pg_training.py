import os
import time
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import BaseCallback
# Stable baselines does NOT have an implementation of REINFORCE directly because A2C replaces it. 
# However, you can approximate it or use their library/contributed repos. We can simulate policy gradients with PPO and A2C.
# But for strict REINFORCE, usually you write it from scratch or rely on other libs.
# Here we will implement PPO and A2C to cover Policy Gradient approaches!

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

def train_pg_model(algorithm=PPO, learning_rate=3e-4, n_steps=2048, ent_coef=0.0, total_timesteps=20000, experiment_name="default_ppo"):
    print(f"--- Training {algorithm.__name__}: {experiment_name} ---")
    log_dir = f"./models/pg/{algorithm.__name__.lower()}/{experiment_name}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Environment Setup
    env = AntFarmEnv(render_mode=None)
    
    # Init Model
    if algorithm == PPO:
        model = PPO(
            "MlpPolicy", 
            env, 
            learning_rate=learning_rate,
            n_steps=n_steps,
            ent_coef=ent_coef,
            verbose=0
        )
    elif algorithm == A2C:
        model = A2C(
            "MlpPolicy", 
            env, 
            learning_rate=learning_rate,
            n_steps=n_steps,
            ent_coef=ent_coef,
            verbose=0
        )

    # Train
    start_time = time.time()
    callback = RewardLoggingCallback()
    model.learn(total_timesteps=total_timesteps, callback=callback)
    
    # Save Model
    model.save(f"{log_dir}/model")
    env.close()
    
    # Plotting Learning Curve
    plt.figure(figsize=(10, 5))
    plt.plot(callback.rewards)
    plt.title(f"{algorithm.__name__} Training - {experiment_name}")
    plt.xlabel('Episodes')
    plt.ylabel('Episode Reward')
    plt.grid(True)
    plt.savefig(f"{log_dir}/learning_curve.png")
    plt.close()
    
    print(f"Finished in {time.time() - start_time:.2f}s.")

if __name__ == "__main__":
    # 10 Experiments for PPO
    train_pg_model(algorithm=PPO, learning_rate=0.0001, ent_coef=0.0, experiment_name="ppo_exp_1")
    train_pg_model(algorithm=PPO, learning_rate=0.0003, ent_coef=0.01, experiment_name="ppo_exp_2")
    train_pg_model(algorithm=PPO, learning_rate=0.0005, ent_coef=0.02, experiment_name="ppo_exp_3")
    train_pg_model(algorithm=PPO, learning_rate=0.0010, ent_coef=0.0, experiment_name="ppo_exp_4")
    train_pg_model(algorithm=PPO, learning_rate=0.0010, ent_coef=0.05, experiment_name="ppo_exp_5")
    train_pg_model(algorithm=PPO, learning_rate=0.0020, ent_coef=0.01, experiment_name="ppo_exp_6")
    train_pg_model(algorithm=PPO, learning_rate=0.0030, ent_coef=0.1, experiment_name="ppo_exp_7")
    train_pg_model(algorithm=PPO, learning_rate=0.0050, ent_coef=0.02, experiment_name="ppo_exp_8")
    train_pg_model(algorithm=PPO, learning_rate=0.0100, ent_coef=0.05, experiment_name="ppo_exp_9")
    train_pg_model(algorithm=PPO, learning_rate=0.0100, ent_coef=0.1, experiment_name="ppo_exp_10")

    # 10 Experiments for A2C
    train_pg_model(algorithm=A2C, learning_rate=0.0001, ent_coef=0.0, experiment_name="a2c_exp_1")
    train_pg_model(algorithm=A2C, learning_rate=0.0005, ent_coef=0.01, experiment_name="a2c_exp_2")
    train_pg_model(algorithm=A2C, learning_rate=0.0010, ent_coef=0.02, experiment_name="a2c_exp_3")
    train_pg_model(algorithm=A2C, learning_rate=0.0020, ent_coef=0.0, experiment_name="a2c_exp_4")
    train_pg_model(algorithm=A2C, learning_rate=0.0030, ent_coef=0.05, experiment_name="a2c_exp_5")
    train_pg_model(algorithm=A2C, learning_rate=0.0050, ent_coef=0.01, experiment_name="a2c_exp_6")
    train_pg_model(algorithm=A2C, learning_rate=0.0070, ent_coef=0.1, experiment_name="a2c_exp_7")
    train_pg_model(algorithm=A2C, learning_rate=0.0100, ent_coef=0.02, experiment_name="a2c_exp_8")
    train_pg_model(algorithm=A2C, learning_rate=0.0500, ent_coef=0.05, experiment_name="a2c_exp_9")
    train_pg_model(algorithm=A2C, learning_rate=0.0500, ent_coef=0.1, experiment_name="a2c_exp_10")
