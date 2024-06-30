"""
deploy.py

Provide a lightweight server/client implementation for deploying OpenVLA models (through the HF AutoClass API) over a
REST API. This script implements *just* the server, with specific dependencies and instructions below.

Note that for the *client*, usage just requires numpy/json-numpy, and requests; example usage below!

Dependencies:
    => Server (runs OpenVLA model on GPU): `pip install uvicorn fastapi json-numpy`
    => Client: `pip install requests json-numpy`

Client (Standalone) Usage (assuming a server running on 0.0.0.0:8000):

```
import requests
import json_numpy
json_numpy.patch()
import numpy as np

action = requests.post(
    "http://0.0.0.0:8000/act",
    json={"image": np.zeros((256, 256, 3), dtype=np.uint8), "instruction": "do something"}
).json()

Note that if your server is not accessible on the open web, you can use ngrok, or forward ports to your client via ssh:
    => `ssh -L 8000:localhost:8000 ssh USER@<SERVER_IP>`
"""

import gymnasium as gym
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from stompy_live.agents.franka_arm import Agent

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import draccus
import torch
import uvicorn
from fastapi import FastAPI
from PIL import Image


# === Server Interface ===
class FrankaServer:
    def __init__(self, model_path: Union[str, Path]) -> Path:
        self.model_path = model_path
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        env_kwargs = dict(obs_mode="state", control_mode="pd_joint_delta_pos", render_mode="human", sim_backend="gpu")
        envs = gym.make("PushCube-v1", **env_kwargs)
        if isinstance(envs.action_space, gym.spaces.Dict):
            envs = FlattenActionSpaceWrapper(envs)
        assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

        agent = Agent(envs).to(device)
        agent.load_state_dict()
        agent.eval()

        self.agent = agent

    def predict_action(self, obs) -> str:
        action = self.agent.get_action(obs)
        return action

    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self.app = FastAPI()
        self.app.post("/act")(self.predict_action)
        uvicorn.run(self.app, host=host, port=port)


@dataclass
class DeployConfig:
    # fmt: off
    model_path: Union[str, Path] = "model.pt"               # Where the model weights are stored

    # Server Configuration
    host: str = "0.0.0.0"                                               # Host IP Address
    port: int = 8000                                                    # Host Port


@draccus.wrap()
def deploy(cfg: DeployConfig) -> None:
    server = FrankaServer(cfg.model_path)
    server.run(cfg.host, port=cfg.port)


if __name__ == "__main__":
    deploy()