from twitch.client import message_queue, init
from threading import Thread
import time, queue
import requests
import gymnasium as gym
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
import torch
import stompy_live.envs.franka_push_cube # noqa: F401

import json_numpy
json_numpy.patch()
import pygame
import numpy as np
import cv2

window = pygame.display.set_mode((1024, 1024))

session = requests.Session()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env_kwargs = dict(obs_mode="rgbd", control_mode="pd_ee_delta_pose", render_mode="rgb_array", sim_backend="gpu")
envs = gym.make("New-PushCube-v1", **env_kwargs)
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
        print(message)
        while not done:
            image = obs["rgb"][0].cpu().numpy()
            now = time.time()

            action = session.post(
                "http://localhost:8000/act",
                json={"image": image, "instruction": message, "unnorm_key": "toto"},
            ).json()

            print("Inference time (ms): ", time.time() - now)

            # For some reason pygame expects the array row/cols to be switched
            transposed_image = np.transpose(image, (1, 0, 2))
            upsized_image = cv2.resize(transposed_image, (1024, 1024), interpolation=cv2.INTER_CUBIC)

            surface = pygame.surfarray.make_surface(upsized_image)
            window.blit(surface, (0, 0))
            
            pygame.display.update()

            obs, reward, terminated, truncated, info = envs.step(torch.tensor(action, dtype=torch.float))

            done = terminated or truncated
            envs.render()
            
    except queue.Empty:
        time.sleep(1)
    except KeyboardInterrupt:
        break