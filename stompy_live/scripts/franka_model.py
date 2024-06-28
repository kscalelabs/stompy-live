"""This module is a proof of concept demonstrating how we can plug in the model from model/franka_arm.py"""

import torch
import gymnasium as gym
from stompy_live.agents.franka_arm import Agent

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env_kwargs = dict(obs_mode="state", control_mode="pd_joint_delta_pos", render_mode="rgb_array", sim_backend="gpu")
envs = gym.make("PushCube-v1", **env_kwargs)
if isinstance(envs.action_space, gym.spaces.Dict):
    envs = FlattenActionSpaceWrapper(envs)
assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

agent = Agent(envs).to(device)
agent.load_state_dict(torch.load("model.pt"))
num_episodes = 10
for episode in range(num_episodes):
    agent.eval()
    obs, info = envs.reset()
    done = False
    total_reward = 0
    
    while not done:
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

envs.close()
