import gymnasium as gym
import torch
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from mani_skill.utils.wrappers.record import RecordEpisode

from stompy_live.agents.franka_arm_vision import Agent


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env_kwargs = dict(obs_mode="rgbd", control_mode="pd_joint_delta_pos", render_mode="rgb_array", sim_backend="gpu")
envs = gym.make("PushCube-v1", **env_kwargs)
envs = FlattenRGBDObservationWrapper(envs, rgb_only=True)
if isinstance(envs.action_space, gym.spaces.Dict):
    envs = FlattenActionSpaceWrapper(envs)
assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

next_obs, _ = envs.reset()
agent = Agent(envs, sample_obs = next_obs).to(device)
agent.load_state_dict(torch.load("model.pt"))
agent.eval()

while True:
    obs, info = envs.reset()
    done = False
    total_reward = 0

    while not done:
        # Get action from the model hosted at the API
        with torch.no_grad():
            action = agent.get_action(obs)
            obs, reward, terminated, truncated, info = envs.step(action)

            done = terminated or truncated
            total_reward += reward
            envs.render()
        envs = RecordEpisode(envs, output_dir="videos", save_trajectory=False)
