"""Client for franka arms that connects to the franka API and sends observations to get actions."""

import argparse
import io

import gymnasium as gym
import requests
import torch
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode

import stompy_live.envs.franka_push_cube # noqa: F401

# Parse franka API route location from command line arguments

parser = argparse.ArgumentParser(description="Client for Franka API")
parser.add_argument("--route", type=str, default="http://localhost:8000/act", help="Where the API is hosted")
args = parser.parse_args()

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env_kwargs = dict(obs_mode="state", control_mode="pd_ee_delta_pose", sim_backend="gpu")

# Create a session for connection reuse
session = requests.Session()

for episode in range(1000):
    env = gym.make("PushCube-v1", **env_kwargs)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    env = RecordEpisode(env, output_dir="eval_videos", save_video=False, video_fps=15, trajectory_name=f"episode_{episode}")
    assert isinstance(env.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    obs, info = env.reset()
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

        obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated
        total_reward += reward
    env.close()
