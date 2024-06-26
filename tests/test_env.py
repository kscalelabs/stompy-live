"""Testing basic maniskill environment using gym."""

import gymnasium as gym
import mani_skill.envs  # noqa: F401
from mani_skill.envs.sapien_env import BaseEnv  # noqa: F401

from stompy_live.agents.stompy import Stompy  # noqa: F401
from stompy_live.envs.stompy_env import StompyEnv  # noqa: F401

env = gym.make(
    "StompyEnv",
    num_envs=1,
    robot_uids="fetch", # test until stompy is fixed
    obs_mode="state",  # there is also "state_dict", "rgbd", ...
    control_mode="pd_ee_delta_pose",  # there is also "pd_joint_delta_pos", ...
    render_mode="human",
)
print("Observation space", env.observation_space)
print("Action space", env.action_space)

obs, _ = env.reset(seed=0)  # reset with a seed for determinism
done = False
infinite = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render()  # a display is required to render
env.close()
