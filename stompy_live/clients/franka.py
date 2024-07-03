"""Client for franka arms that connects to the franka API and sends observations to get actions."""

import argparse
import io

import gymnasium as gym
import requests
import torch
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper

# Parse franka API route location from command line arguments

parser = argparse.ArgumentParser(description="Client for Franka API")
parser.add_argument("--route", type=str, default="http://localhost:8000/act", help="Where the API is hosted")
args = parser.parse_args()

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env_kwargs = dict(obs_mode="state", control_mode="pd_joint_delta_pos", render_mode="human", sim_backend="gpu")
envs = gym.make("PushCube-v1", **env_kwargs)
if isinstance(envs.action_space, gym.spaces.Dict):
    envs = FlattenActionSpaceWrapper(envs)
assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

# Create a session for connection reuse
session = requests.Session()

while True:
    obs, info = envs.reset()
    print(obs)
    done = False
    total_reward = 0

    while not done:
        # Get action from the model hosted at the API
        with torch.no_grad():
            buffer = io.BytesIO()
            torch.save(obs, buffer)
            obs_bytes = buffer.getvalue()

            action_bytes = session.post(args.route, data=obs_bytes).content
            action = torch.load(io.BytesIO(action_bytes))

        obs, reward, terminated, truncated, info = envs.step(action)

        done = terminated or truncated
        total_reward += reward
        envs.render()
