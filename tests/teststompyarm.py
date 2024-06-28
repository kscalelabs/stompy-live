"""Testing basic maniskill environment using gym."""

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np  # noqa: F401
from mani_skill.envs.sapien_env import BaseEnv  # noqa: F401

from stompy_live.agents.stompy.stompy import Stompy  # noqa: F401
from stompy_live.envs.panda_env import PandaPushCubeEnv  # noqa: F401
from stompy_live.envs.stompy_env import StompyEnv  # noqa: F401
from stompy_live.envs.stompyarm_env import StompyPushCubeEnv  # noqa: F401

# model = model_client()
language_instruction = "move the apple to the right"

env = gym.make(
    "SPushCube-v0",
    num_envs=1,
    # robot_uids="stompy_arm",  # test until stompy is fixed
    obs_mode="state",  # there is also "state_dict", "rgbd", ...
    control_mode="pd_joint_delta_pos",  # there is also "pd_joint_delta_pos", ...
    render_mode="human",
)
print("Observation space", env.observation_space)
print("Action space", env.action_space)

obs, _ = env.reset(seed=0)  # reset with a seed for determinism
done = False
infinite = False
while not infinite:
    # grpc or websocket
    # action = model.act(obs["image", "qpos", "language_instruction"])
    action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # action = env.action_space.sample()
    # print(action)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render()  # a display is required to render
env.close()
