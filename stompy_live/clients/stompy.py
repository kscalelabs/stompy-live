"""Stompy client. Just performs random actions for now."""

import gymnasium as gym
import torch
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from stompy_live.envs.stompy_env import SceneManipulationEnv # noqa: F401

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env_kwargs = dict(obs_mode="state", control_mode="pd_joint_delta_pos", render_mode="human", sim_backend="gpu")
env = gym.make("New-SceneManipulation-v1", **env_kwargs, scene_builder_cls="ai2thor")
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
        env.render()
