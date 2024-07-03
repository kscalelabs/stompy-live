import gymnasium as gym
import torch
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode

from stompy_live.agents.franka.franka_arm import Agent
import stompy_live.envs.franka_push_cube # noqa: F401

model_path = "model.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env_kwargs = dict(obs_mode="state", control_mode="pd_ee_delta_pose", sim_backend="gpu")
env = gym.make("New-PushCube-v1", **env_kwargs)
if isinstance(env.action_space, gym.spaces.Dict):
    env = FlattenActionSpaceWrapper(env)
assert isinstance(env.single_action_space, gym.spaces.Box), "only continuous action space is supported"

agent = Agent(env).to(device)
agent.load_state_dict(torch.load(model_path))
agent.eval()

for episode in range(1000):
    env = RecordEpisode(env, output_dir="data", save_video=False, trajectory_name=f"episode_{episode}")
    obs, info = env.reset()
    print(obs)
    done = False
    total_reward = 0

    while not done:
        # Get action from the model hosted at the API
        with torch.no_grad():
            action = agent.get_action(obs)

        obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated
        total_reward += reward