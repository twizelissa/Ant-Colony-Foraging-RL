# Ant Colony Foraging Optimizer (Mission Based RL)

This repository contains an RL solution designed to manage a simulated ant colony foraging environment.

**Video Demonstration:** [Watch the trained agent in action](https://www.bugufi.link/0gSh2k)

**Full Report:** [Summative Report](Summative_Report_Final.pdf)

## 🐜 Mission Description 
The agent manages an ant colony by controlling the pheromone levels along different foraging trails. The **objective** is to bring the maximum amount of food back to the nest efficiently.

- **State Space**: Normalized food levels remaining, current pheromone intensities along paths, distance values, and normalized time remaining.
- **Action Space**: Discrete actions to boost pheromones on the trails leading to the 3 food sources, or do nothing.
- **Rewards**: +1 for food collected, with minor energy penalties applied when wasting resources on dropping unnecessary pheromones.

## 📁 Directory Structure
- `environment/custom_env.py` - Contains the Gymnasium environment and Pygame visualizer.
- `training/dqn_training.py` - Contains 10 runs testing DQN hyperparameters.
- `training/pg_training.py` - Contains 10 runs testing PPO and 10 runs testing A2C. 
- `main.py` - Central entry point to run random agents or playback trained agents visually.

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
