import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from environment.custom_env import AntFarmEnv

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        logits = self.fc(x)
        return self.softmax(logits)

def train_reinforce(learning_rate=1e-3, gamma=0.99, episodes=500, experiment_name="reinforce_exp"):
    print(f"--- Training REINFORCE: {experiment_name} ---")
    log_dir = f"./models/pg/reinforce/{experiment_name}"
    os.makedirs(log_dir, exist_ok=True)

    env = AntFarmEnv(render_mode=None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = PolicyNetwork(state_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

    all_rewards = []
    start_time = time.time()

    for episode in range(episodes):
        state, _ = env.reset()
        log_probs = []
        rewards = []

        done = False
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            probs = policy(state_tensor)
            m = Categorical(probs)
            action = m.sample()

            next_state, reward, done, _, info = env.step(action.item())

            log_probs.append(m.log_prob(action))
            rewards.append(reward)
            state = next_state

        # Compute discounted returns
        returns = []
        R = 0
        for r in rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
            
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * R)
            
        optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        optimizer.step()

        total_reward = sum(rewards)
        all_rewards.append(total_reward)

    plt.figure(figsize=(10, 5))
    plt.plot(all_rewards)
    plt.title(f"REINFORCE Training - {experiment_name}")
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.savefig(f"{log_dir}/learning_curve.png")
    plt.close()
    
    torch.save(policy.state_dict(), f"{log_dir}/model.pth")
    print(f"Finished in {time.time() - start_time:.2f}s. Avg Reward: {sum(all_rewards[-10:])/10:.2f}")

if __name__ == "__main__":
    train_reinforce(learning_rate=0.0001, gamma=0.99, experiment_name="reinforce_exp_1")
    train_reinforce(learning_rate=0.0005, gamma=0.99, experiment_name="reinforce_exp_2")
    train_reinforce(learning_rate=0.0010, gamma=0.95, experiment_name="reinforce_exp_3")
    train_reinforce(learning_rate=0.0010, gamma=0.99, experiment_name="reinforce_exp_4")
    train_reinforce(learning_rate=0.0020, gamma=0.90, experiment_name="reinforce_exp_5")
    train_reinforce(learning_rate=0.0020, gamma=0.99, experiment_name="reinforce_exp_6")
    train_reinforce(learning_rate=0.0050, gamma=0.95, experiment_name="reinforce_exp_7")
    train_reinforce(learning_rate=0.0050, gamma=0.99, experiment_name="reinforce_exp_8")
    train_reinforce(learning_rate=0.0100, gamma=0.90, experiment_name="reinforce_exp_9")
    train_reinforce(learning_rate=0.0100, gamma=0.99, experiment_name="reinforce_exp_10")
