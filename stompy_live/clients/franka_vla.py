from twitch.client import message_queue, init
from threading import Thread
import time, queue
import requests
import gymnasium as gym
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
import torch

import json_numpy
json_numpy.patch()
import numpy as np

session = requests.Session()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env_kwargs = dict(obs_mode="rgbd", control_mode="pd_ee_delta_pose", render_mode="rgb_array", sim_backend="gpu")
envs = gym.make("PushCube-v1", **env_kwargs)
envs = FlattenRGBDObservationWrapper(envs, rgb_only=True)
if isinstance(envs.action_space, gym.spaces.Dict):
    envs = FlattenActionSpaceWrapper(envs)
assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"


# Initializes Twitch IRC thread
irc_thread = Thread(target=init)
irc_thread.daemon = True  # This allows the thread to exit when the main program does
irc_thread.start()

while True:
    obs, info = envs.reset()
    done = False

    try:
        message = message_queue.get(block=False)
        while not done:
            action = session.post(
                "http://localhost:8000/act",
                json={"image": obs["state"].cpu().numpy(), "instruction": message, "unnorm_key": "toto"}
            ).json()
            print(action)

            obs, reward, terminated, truncated, info = envs.step(torch.tensor(action, dtype=torch.float))

            done = terminated or truncated
            envs.render()
            envs = RecordEpisode(envs, output_dir="videos", save_trajectory=False)
    except queue.Empty:
        time.sleep(1)
    except KeyboardInterrupt:
        break