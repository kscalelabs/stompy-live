import torch
import torch.nn as nn
import mani_skill
import mani_skill.envs
import numpy as np
import gymnasium as gym

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Actor mean network
        self.actor_mean = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        
        # Log standard deviation of the actor
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, obs):
        value = self.critic(obs)
        action_mean = self.actor_mean(obs)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        return action_mean, action_std, value

    def act(self, obs):
        action_mean, action_std, _ = self(obs)
        action = torch.normal(action_mean, action_std)
        return action

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the environment to get observation and action dimensions
env = gym.make("PickCube-v1", obs_mode="state", control_mode="pd_joint_pos")
obs_dim = env.observation_space['state'].shape[0]  # Assuming 'state' is the key for proprioceptive state
action_dim = env.action_space.shape[0]

# Initialize the model and load state dict
model = ActorCritic(obs_dim, action_dim).to(device)
state_dict = torch.load("model.pt", map_location=device)
model.load_state_dict(state_dict)
model.eval()

# Run the model in the environment
num_episodes = 10
for episode in range(num_episodes):
    obs, info = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        # Extract the state from the observation dictionary
        state = obs['state']
        
        # Preprocess the observation
        obs_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        # Get action from the model
        with torch.no_grad():
            action = model.act(obs_tensor)
        
        # Convert action to numpy and execute in the environment
        action_np = action.cpu().numpy().squeeze()
        obs, reward, terminated, truncated, info = env.step(action_np)
        
        done = terminated or truncated
        total_reward += reward
    
    print(f"Episode {episode + 1} finished with reward: {total_reward}")

env.close()
