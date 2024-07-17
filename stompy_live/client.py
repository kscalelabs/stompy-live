"""Stompy client. Just performs random actions for now."""

import argparse
import subprocess

import gymnasium as gym
import torch
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper

from stompy_live.envs.stompy_env import SceneManipulationEnv  # noqa: F401

parser = argparse.ArgumentParser(description="Client that simulates Stompy")
parser.add_argument(
    "--streamkey",
    type=str,
    default=None,
    help="Streamer key for streaming to Twitch. Passing in this argument will stream the output to Twitch.",
)
args = parser.parse_args()

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.streamkey is not None:
    render_mode = "rgb_array"
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-r",
        "15",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        "512x512",
        "-re",  # for real-time output
        "-i",
        "-",  # The input comes from a pipe
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "veryfast",
        "-b:v",
        "3000k",  # Set video bitrate
        "-bufsize",
        "6000k",  # Set buffer size
        "-f",
        "flv",
        f"rtmp://live.twitch.tv/app/{args.streamkey}",
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
else:
    render_mode = "human"
env_kwargs = dict(obs_mode="state", control_mode="pd_joint_delta_pos", render_mode=render_mode, sim_backend="gpu")
env = gym.make("New-SceneManipulation-v1", **env_kwargs, robot_uids="fetch")
if isinstance(env.action_space, gym.spaces.Dict):
    env = FlattenActionSpaceWrapper(env)
assert isinstance(env.single_action_space, gym.spaces.Box), "only continuous action space is supported"

time = 0

while True:
    obs, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        # Get action from the model hosted at the API
        time += 1
        with torch.no_grad():
            if time > 30 and time % 13 == 0:
                action = env.action_space.sample() * 100
            elif time <= 30:
                action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated
        total_reward += reward
        if args.streamkey is not None:
            image = env.render().cpu().numpy()[0]
            process.stdin.write(image.tobytes())
        else:
            env.render()
