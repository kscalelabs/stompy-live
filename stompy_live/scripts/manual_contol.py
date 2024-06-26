"""Keyboard override for testing purposes."""

import argparse
import logging
import sys
import termios
import tty

import tkinter as tk
import gymnasium as gym
import numpy as np
from mani_skill.utils.wrappers import RecordEpisode

from simgame.envs.stompy_arm import StompyPushCubeEnv

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--env-id",
        type=str,
        default="StompyPushCube-v1",
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
        help=(
            "Change shader used for rendering. Default is 'default' which is very fast. Can also be 'rt' for ray "
            "tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality "
            "ray-traced renderer"
        ),
    )
    parser.add_argument("--record-dir", type=str)
    parser.add_argument(
        "-p",
        "--pause",
        action="store_true",
        help="If using human render mode, auto pauses the simulation upon loading",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable verbose output.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        help="Seed the random actions and simulator. Default is no seed",
    )
    args = parser.parse_args()

    return args


def get_keypress() -> str:
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        key = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return key


def get_key(root: any) -> str:
    key_pressed = []

    def on_key_press(event: any) -> None:
        key_pressed.append(event.keysym)
        root.quit()  # Close the tkinter main loop

    root.bind("<KeyPress>", on_key_press)
    root.mainloop()


def main() -> None:
    np.set_printoptions(suppress=True, precision=3)

    args = parse_args()
    logging.basicConfig(level=logging.INFO if args.quiet else logging.DEBUG)

    if args.seed is not None:
        np.random.seed(args.seed)

    env: StompyPushCubeEnv = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        render_mode=args.render_mode,
        shader_dir=args.shader,
        sim_backend=args.sim_backend,
    )

    record_dir = args.record_dir
    if record_dir:
        record_dir = record_dir.format(env_id=args.env_id)
        env = RecordEpisode(env, record_dir, info_on_video=True)

    logger.debug("Observation space: %s", env.observation_space)
    logger.debug("Action space: %s", env.action_space)
    logger.debug("Control mode: %s", env.unwrapped.control_mode)
    logger.debug("Reward mode: %s", env.unwrapped.reward_mode)

    obs, _ = env.reset(seed=args.seed)
    logger.debug("Initial observation: %s", obs)

    env.action_space.seed(args.seed)
    if args.render_mode is not None:
        viewer = env.render()
        viewer.paused = args.pause
        env.render()

    # action_1 = np.array([-0.912, -0.639, 0.154, 0.55, 0.693, 1.0], dtype=np.float32)
    # action_2 = np.array([-0.912, -0.639, 0.124, 0.55, 0.693, 1.0], dtype=np.float32)
    # action_1_flag = False

    if (action_space_size := env.action_space.shape) is None:
        raise ValueError("The action space size is not defined.")

    key_to_action = [
        ("q", "a"),
        ("w", "s"),
        ("e", "d"),
        ("r", "f"),
        ("t", "g"),
        ("y", "h"),
        ("u", "j"),
        ("i", "k"),
        ("o", "l"),
        ("p", ";"),
    ]

    if len(action_space_size) > len(key_to_action):
        raise ValueError("Too many actions to map to keys.")

    root = tk.Tk()
    root.withdraw()

    while True:
        # key = get_keypress()
        key = get_key(root)
        # Gets the action array based on the keyboard inpts.
        action = np.zeros(action_space_size)
        if key != -1:
            for i in range(len(action)):
                left_key, right_key = key_to_action[i]
                if key == left_key:
                    action[i] = 1
                elif key == right_key:
                    action[i] = -1

        obs, reward, terminated, truncated, info = env.step(action)

        logger.debug("Action Taken: %s", action)
        logger.debug("Reward: %s", reward)
        logger.debug("Observation: %s", obs)
        logger.debug("Terminated: %s", terminated)
        logger.debug("Info: %s", info)

        if args.render_mode is not None:
            env.render()

        if args.render_mode is None or args.render_mode != "human":
            if terminated or truncated:
                break
    env.close()

    if record_dir is not None:
        logger.info("Saving video to %s", record_dir)


if __name__ == "__main__":
    # python -m simgame.scripts.control_robot
    main()
