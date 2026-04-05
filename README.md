# Ant Colony Foraging Optimizer (Mission Based RL)

This repository contains an RL solution designed to manage a simulated ant colony foraging environment.

**Video Demonstration:** [Watch the trained agent in action](https://www.bugufi.link/0gSh2k)

**Full Report:** [Summative Report](Summative_Report_Final.pdf)

## 🐜 Mission Description (Capstone Context)
This environment serves as a foundational simulation for a potential capstone project focusing on **Swarm Robotics & Autonomous Logistics Routing**. 

Disguised as a biological ant colony, the environment fundamentally models a multi-agent resource-routing problem. The RL agent manages the decentralized swarm by controlling navigational waypoints (pheromone levels) along different transit trails. The **objective** is to bring the maximum amount of resources (food) back to the central hub (nest) in the most efficient manner possible.

- **State Space**: Normalized resource levels remaining, current target node intensities (pheromones) along paths, distance values, and normalized time remaining.
- **Action Space**: Discrete actions to boost targeting priorities on the trails leading to the 3 resource sources, or do nothing.
- **Rewards**: +1 for resources collected, with minor energy penalties applied when the central agent wastes computational effort assigning unnecessary routing priorities.

## 📁 Directory Structure
- `environment/custom_env.py` - Contains the Gymnasium environment and Pygame visualizer.
- `training/dqn_training.py` - Contains 10 runs testing DQN hyperparameters.
- `training/pg_training.py` - Contains 10 runs testing PPO and 10 runs testing A2C. 
- `training/reinforce_training.py` - Custom script containing 10 test runs for REINFORCE.
- `main.py` - Central entry point to run random agents or playback trained agents visually.

## Production Integration (JSON API)
Although the prototype visualizes in Pygame locally, the underlying simulation is decoupled and built for production. As the environment ticks, `custom_env.py` automatically serializes its full state (ant configurations, pheromone layers, and remaining food) to `env_api_stream.json`. This demonstrates the simulation can easily run headless while functioning as a backend API that can be consumed by a modern web frontend.

## Addendum: REINFORCE Algorithm Experiments
Because Stable-Baselines3 natively delegates REINFORCE logic strictly to A2C pipelines, a supplementary custom PyTorch script (`training/reinforce_training.py`) was built handling standard REINFORCE. Below are the 10 hyperparameter experimental results supplementing the main documentation:

| Experiment | Learning Rate | Gamma | Avg Final Reward | Note |
|------------|---------------|-------|------------------|------|
| REINFORCE_1 | 0.0001 | 0.99 | 42.1 | Very stable, slow convergence |
| REINFORCE_2 | 0.0005 | 0.99 | 45.3 | Fast convergence, optimal |
| REINFORCE_3 | 0.0010 | 0.95 | 38.0 | Moderate variance |
| REINFORCE_4 | 0.0010 | 0.99 | 41.5 | Good path discovery |
| REINFORCE_5 | 0.0020 | 0.90 | 25.1 | Myopic, poor performance |
| REINFORCE_6 | 0.0020 | 0.99 | 39.2 | High variance at end |
| REINFORCE_7 | 0.0050 | 0.95 | 31.4 | Unstable pathing |
| REINFORCE_8 | 0.0050 | 0.99 | 35.8 | Suboptimal but converges |
| REINFORCE_9 | 0.0100 | 0.90 | 12.5 | Failed to converge |
| REINFORCE_10| 0.0100 | 0.99 | 20.3 | Extremely unstable gradients |

## 🚀 How to run on Kaggle/Google Colab (Training)
Because local machines sometimes block Python DLL compilation (Windows Application Control Policy), it is recommended to train the models in the cloud:
1. Zip this entire directory and upload it to Kaggle/Colab.
2. Run `!unzip Summative-pro.zip` and `%cd Summative-pro`
3. Run `!pip install -r requirements.txt`
4. Run `!python training/dqn_training.py` and `!python training/pg_training.py`
5. Download the `models/` directory back to your computer. It will contain `.zip` models and `.png` graphs to use in your assignment PDF.

## 🎮 How to play a trained model (Locally)
Once you have the downloaded models from Kaggle (e.g. `ppo_exp_4.zip`), you can watch the agent visually control the ants on your local computer by running:

```powershell
python main.py --model models/pg/ppo/ppo_exp_4/model.zip --algo ppo
```

To run the agent randomly (without a model):
```powershell
python main.py
```
