import gymnasium as gym
import torch
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper

from stompy_live.agents.franka.franka_arm import Agent
import stompy_live.envs.franka_push_cube # noqa: F401

import json
from tqdm import tqdm
model_path = "runs/New-PushCube-v1__ppo__1__1720423824/final_ckpt.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env_kwargs = dict(obs_mode="state", control_mode="pd_ee_delta_pose", sim_backend="gpu", render_mode="rgb_array")

for episode in tqdm(range(10000)):
    env = gym.make("New-PushCube-v1", **env_kwargs)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    assert isinstance(env.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(env).to(device)
    agent.load_state_dict(torch.load(model_path))
    agent.eval()
    obs, info = env.reset()
    total_reward = 0

    success = False

    episode_data = []
    for step in range(10**10):
        image = env.render()
        
        with torch.no_grad():
            action = agent.get_action(obs)

        episode_data.append({
            "image": image.cpu().numpy()[0].tolist(),
            "action": action.cpu().numpy().tolist()
        })

        obs, _, terminated, truncated, info = env.step(action)

        if terminated:
            success = True
        
        if terminated or truncated:
            break
    
    if success:
        json_data = json.dumps(episode_data)
        with open(f'data/episode-{episode}.json', 'w') as f:
            f.write(json_data)