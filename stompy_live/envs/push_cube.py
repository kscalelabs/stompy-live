"""Defines an environment for controlling the Stompy arm."""

from typing import Any

import numpy as np
import torch
import torch.random
from mani_skill.envs.tasks.tabletop.push_cube import PushCubeEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import Array
from torch import Tensor
from transforms3d.euler import euler2quat

from stompy_live.agents.stompyarm.stompyarm import StompyArm
from stompy_live.utils.scene_builders.table_builder import StompyTableSceneBuilder


@register_env("SPushCube-v0", max_episode_steps=50)
class StompyPushCubeEnv(PushCubeEnv):
    SUPPORTED_ROBOTS = ["stompy_arm"]

    agent: StompyArm

    # Set some commonly used values
    goal_radius = 0.1
    cube_half_size = 0.02

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        super().__init__(*args, robot_uids="stompy_arm", **kwargs)

    @property
    def _default_sensor_configs(self) -> list[CameraConfig]:
        # Registers one 128x128 camera looking at the robot, cube, and target
        # a smaller sized camera will be lower quality, but render faster
        pose = sapien_utils.look_at(eye=[0.3, -0.3, 0.6], target=[-0.1, 0, 0.1])
        return [
            CameraConfig(
                "base_camera",
                pose=pose,
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
            )
        ]

    @property
    def _default_human_render_camera_configs(self) -> CameraConfig:
        # Registers a more high-definition (512x512) camera used just for
        # rendering when render_mode="rgb_array" or calling
        # env.render_rgb_array()
        pose = sapien_utils.look_at([0.9, 0, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose=pose, width=512, height=512, fov=1.5, near=0.01, far=100)

    def _load_scene(self, options: dict) -> None:
        # We use a prebuilt scene builder class that automatically loads in a
        # floor and table.
        self.table_scene = StompyTableSceneBuilder(env=self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()

        # We then add the cube that we want to push and give it a color and
        # size using a convenience build_cube function we specify the body_type
        # to be "dynamic" as it should be able to move when touched by other
        # objects / the robot
        self.obj = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=np.array([12, 42, 160, 255]) / 255,
            name="cube",
            body_type="dynamic",
        )

        # We also add in red/white target to visualize where we want the cube
        # to be pushed to we specify add_collisions=False as we only use this
        # as a visual for videos and do not want it to affect the actual physics
        # we finally specify the body_type to be "kinematic" so that the object
        # stays in place
        self.goal_region = actors.build_red_white_target(
            self.scene,
            radius=self.goal_radius,
            thickness=1e-5,
            name="goal_region",
            add_collision=False,
            body_type="kinematic",
        )

    def _initialize_episode(self, env_idx: Tensor, options: dict) -> None:
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            xyz = torch.zeros((b, 3))
            xyz[..., :2] = torch.rand((b, 2)) * 0.2 - 0.1
            xyz[..., 2] = self.cube_half_size
            xyz = xyz + torch.tensor([0.1, 0.15, 0])
            q = [1, 0, 0, 0]
            obj_pose = Pose.create_from_pq(p=xyz, q=q)
            self.obj.set_pose(obj_pose)

            target_region_xyz = xyz + torch.tensor([0.1 + self.goal_radius, 0, 0])
            target_region_xyz[..., 2] = 1e-3
            target_region_xyz = target_region_xyz + torch.tensor([0, -0.05, 0])
            self.goal_region.set_pose(
                Pose.create_from_pq(
                    p=target_region_xyz,
                    q=euler2quat(0, np.pi / 2, 0),
                )
            )

    def evaluate(self) -> dict[str, Tensor]:
        cur_radius = torch.linalg.norm(self.obj.pose.p[..., :2] - self.goal_region.pose.p[..., :2], axis=1)
        is_obj_placed = cur_radius < self.goal_radius

        return {
            "success": is_obj_placed,
        }

    def compute_dense_reward(self, obs: Any, action: Array, info: dict) -> Tensor:  # noqa: ANN401
        # We also create a pose marking where the robot should push the cube from that is easiest (pushing from behind the cube)
        # tcp_push_pose = Pose.create_from_pq(
        #     p=self.obj.pose.p
        #     + torch.tensor([-self.cube_half_size - 0.005, 0, 0], device=self.device)
        # )
        # tcp_to_push_pose = tcp_push_pose.p - self.agent.tcp.pose.p
        # tcp_to_push_pose_dist = torch.linalg.norm(tcp_to_push_pose, axis=1)
        # reaching_reward = 1 - torch.tanh(5 * tcp_to_push_pose_dist)
        # reward = reaching_reward

        # # compute a placement reward to encourage robot to move the cube to the center of the goal region
        # # we further multiply the place_reward by a mask reached so we only add the place reward if the robot has reached the desired push pose
        # # This reward design helps train RL agents faster by staging the reward out.
        # reached = tcp_to_push_pose_dist < 0.01
        # obj_to_goal_dist = torch.linalg.norm(
        #     self.obj.pose.p[..., :2] - self.goal_region.pose.p[..., :2], axis=1
        # )
        # place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        # reward += place_reward * reached

        # reward for moving cube near
        print(self.obj.pose.p[..., :2])
        obj_to_goal_dist = torch.linalg.norm(
            self.obj.pose.p[..., :2] - self.goal_region.pose.p[..., :2], axis=1
        )

        # reward for gripper near cube
        gripper_to_cube_dist = torch.linalg.norm(
            self.obj.pose.p[..., :2] - self.agent.ee.pose.p[..., :2], axis=1
        )

        print(obj_to_goal_dist)
        print(gripper_to_cube_dist)

        reward = 0.1 * obj_to_goal_dist + 0.01 * gripper_to_cube_dist
        print(reward)

        # assign rewards to parallel environments that achieved success to the maximum of 3.
        reward[info["success"]] = 3
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: dict) -> Tensor:  # noqa: ANN401
        # This should be equal to compute_dense_reward / max possible reward
        max_reward = 3.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
