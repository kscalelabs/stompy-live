# use python -m mani_skill.examples.demo_stompy --record-dir="videos" to save video

import argparse

import gymnasium as gym
import numpy as np

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.wrappers import RecordEpisode

from typing import Any, Dict, Union
import os
import torch
import torch.random
from transforms3d.euler import euler2quat

from mani_skill.agents.robots import StompyArm
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.controllers import *

from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--env-id",
        type=str,
        default="Stompy-PushCube",
        help="The environment ID of the task you want to simulate",
    )
    parser.add_argument("-o", "--obs-mode", type=str, default="none")
    parser.add_argument(
        "-b",
        "--sim-backend",
        type=str,
        default="auto",
        help="Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'",
    )
    parser.add_argument("--reward-mode", type=str)
    parser.add_argument("-c", "--control-mode", type=str)
    parser.add_argument("--render-mode", type=str, default="rgb_array")
    parser.add_argument(
        "--shader",
        default="default",
        type=str,
        help="Change shader used for rendering. Default is 'default' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer",
    )
    parser.add_argument("--record-dir", type=str)
    parser.add_argument(
        "-p",
        "--pause",
        action="store_true",
        help="If using human render mode, auto pauses the simulation upon loading",
    )
    parser.add_argument("--quiet", action="store_true", help="Disable verbose output.")
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        help="Seed the random actions and simulator. Default is 0",
        default=0,
    )
    args, opts = parser.parse_known_args(args)

    # Parse env kwargs
    if not args.quiet:
        print("opts:", opts)
    eval_str = lambda x: eval(x[1:]) if x.startswith("@") else x
    env_kwargs = dict((x, eval_str(y)) for x, y in zip(opts[0::2], opts[1::2]))
    if not args.quiet:
        print("env_kwargs:", env_kwargs)
    args.env_kwargs = env_kwargs

    return args


def main(args):
    # print which link is the root
    # print(StompyArm.urdf_config)
    # print all links from arm
    # print(StompyArm.fix_root_link)
    # breakpoint()
    # print(StompyArm.get_state(StompyArm))
    # print(StompyArm.get_proprioception())
    np.set_printoptions(suppress=True, precision=3)
    verbose = not args.quiet
    if args.seed is not None:
        np.random.seed(args.seed)
    env: BaseEnv = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        render_mode=args.render_mode,
        shader_dir=args.shader,
        sim_backend=args.sim_backend,
        **args.env_kwargs,
    )

    record_dir = args.record_dir
    if record_dir:
        record_dir = record_dir.format(env_id=args.env_id)
        env = RecordEpisode(env, record_dir, info_on_video=True)

    if verbose:
        print("Observation space", env.observation_space)
        print("Action space", env.action_space)
        print("Control mode", env.unwrapped.control_mode)
        print("Reward mode", env.unwrapped.reward_mode)

    obs, _ = env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    if args.render_mode is not None:
        viewer = env.render()
        viewer.paused = args.pause
        env.render()

    while True:
        action = env.action_space.sample()
        print(action)
        # action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # print(action)
        obs, reward, terminated, truncated, info = env.step(action)
        if verbose:
            # print("reward", reward)
            # print("terminated", terminated)
            # print("truncated", truncated)
            # print("info", info)
            # print("observation", obs)
            # print(" ".join(f"{key}: {value}" for key, value in obs.items()))
            pass

        if args.render_mode is not None:
            env.render()

        if args.render_mode is None or args.render_mode != "human":
            if terminated or truncated:
                print("Resetting environment")
                break
    env.close()

    if record_dir:
        print(f"Saving video to {record_dir}")


if __name__ == "__main__":
    main(parse_args())
