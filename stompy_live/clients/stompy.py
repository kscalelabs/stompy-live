"""Stompy client. Just performs random actions for now."""

import argparse
import cv2
import subprocess
from typing import Optional

parser = argparse.ArgumentParser(description="Client that simulates Stompy")
parser.add_argument("--streamkey", type=str, default=None, help="Streamer key for streaming to Twitch. Passing in this argument will stream the output to Twitch.")
args = parser.parse_args()

import gymnasium as gym
import torch
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper

from stompy_live.envs.stompy_env import SceneManipulationEnv  # noqa: F401

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.streamkey is not None:
    render_mode = "rgb_array"
    command = [
        'ffmpeg',
        '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'rgb24',
        '-s', '512x512',
        '-re', # for real-time output
        '-i', '-',  # The input comes from a pipe
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'veryfast',
        '-f', 'flv',
        f'rtmp://live.twitch.tv/app/{args.streamkey}'
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
else:
    render_mode = "human"
env_kwargs = dict(obs_mode="state", control_mode="pd_joint_delta_pos", render_mode=render_mode, sim_backend="gpu")
env = gym.make("New-SceneManipulation-v1", **env_kwargs)
if isinstance(env.action_space, gym.spaces.Dict):
    env = FlattenActionSpaceWrapper(env)
assert isinstance(env.single_action_space, gym.spaces.Box), "only continuous action space is supported"

while True:
    obs, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        # Get action from the model hosted at the API
        with torch.no_grad():
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated
        total_reward += reward
        if args.streamkey is not None:
            image = env.render().cpu().numpy()[0]
            process.stdin.write(image.tobytes())
        else:
            env.render()
