"""This module is a proof of concept demonstrating how we can plug in the model from model/franka_arm.py"""

import gymnasium as gym
import torch
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper

from stompy_live.agents.franka_arm import Agent

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env_kwargs = dict(obs_mode="state", control_mode="pd_joint_delta_pos", render_mode="human", sim_backend="gpu")
envs = gym.make("PushCube-v1", **env_kwargs)
if isinstance(envs.action_space, gym.spaces.Dict):
    envs = FlattenActionSpaceWrapper(envs)
assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

agent = Agent(envs).to(device)
agent.load_state_dict(torch.load("model.pt"))
while True:
    agent.eval()
    obs, info = envs.reset()
    done = False
    total_reward = 0

    while not done:
        # Get action from the model
        with torch.no_grad():
            action = agent.get_action(obs)

        obs, reward, terminated, truncated, info = envs.step(action)

        done = terminated or truncated
        total_reward += reward
        envs.render()
