import argparse
import sys
import time
from environment.custom_env import AntFarmEnv

# Attempt to import stable-baselines3, but don't fail if the user is just running random.
try:
    from stable_baselines3 import DQN, PPO, A2C
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

def run_random():
    env = AntFarmEnv(render_mode="human")
    obs, info = env.reset()
    done = False
    truncated = False
    print("Running random agent in Ant Colony Environment...")
    while not (done or truncated):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        time.sleep(0.01) 
    print(f"Simulation finished! Total food collected: {info.get('collected')}")
    env.close()

def run_trained_model(model_path, algo_type):
    if not SB3_AVAILABLE:
        print("Error: stable-baselines3 must be installed to run trained models.")
        sys.exit(1)
        
    print(f"Loading {algo_type} model from {model_path}...")
    
    if algo_type.lower() == "dqn":
        model = DQN.load(model_path)
    elif algo_type.lower() == "ppo":
        model = PPO.load(model_path)
    elif algo_type.lower() == "a2c":
        model = A2C.load(model_path)
    else:
        print("Unknown algorithm type. Use dqn, ppo, or a2c.")
        sys.exit(1)

    env = AntFarmEnv(render_mode="human")
    obs, info = env.reset()
    done = False
    truncated = False

    while not (done or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(int(action))
        time.sleep(0.01)

    print(f"Simulation finished! Total food collected by trained agent: {info.get('collected')}")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ant Colony Optimizer")
    parser.add_argument("--model", type=str, help="Path to the trained model zip file")
    parser.add_argument("--algo", type=str, default="ppo", help="Algorithm type: dqn, ppo, or a2c")
    args = parser.parse_args()

    if args.model:
        run_trained_model(args.model, args.algo)
    else:
        run_random()
