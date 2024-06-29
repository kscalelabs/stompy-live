import io
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import draccus
import gymnasium as gym
import torch
import uvicorn
from fastapi import FastAPI, Request, Response
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper

from stompy_live.agents.franka_arm import Agent


# === Server Interface ===
class FrankaServer:
    def __init__(self, model_path: Union[str, Path]) -> Path:
        self.model_path = model_path
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        env_kwargs = dict(obs_mode="rgbd", control_mode="pd_joint_delta_pos", render_mode="human", sim_backend="gpu")
        envs = gym.make("PushCube-v1", **env_kwargs)
        if isinstance(envs.action_space, gym.spaces.Dict):
            envs = FlattenActionSpaceWrapper(envs)
        assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

        agent = Agent(envs).to(device)
        agent.load_state_dict(torch.load(model_path))
        agent.eval()

        self.agent = agent

    async def predict_action(self, request: Request) -> Response:
        obs_bytes = await request.body()
        obs_tensor = torch.load(io.BytesIO(obs_bytes))
        action = self.agent.get_action(obs_tensor)

        # Return as bytes
        buffer = io.BytesIO()
        torch.save(action, buffer)
        action_bytes = buffer.getvalue()

        return Response(content=action_bytes, media_type="application/octet-stream")

    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self.app = FastAPI()
        self.app.post("/act")(self.predict_action)
        uvicorn.run(self.app, host=host, port=port)


@dataclass
class DeployConfig:
    # fmt: off
    model_path: Union[str, Path] = "model-rgb.pt"               # Where the model weights are stored

    # Server Configuration
    host: str = "0.0.0.0"                                               # Host IP Address
    port: int = 8000                                                    # Host Port


@draccus.wrap()
def deploy(cfg: DeployConfig) -> None:
    server = FrankaServer(cfg.model_path)
    server.run(cfg.host, port=cfg.port)


if __name__ == "__main__":
    deploy()
