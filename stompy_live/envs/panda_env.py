"""Tabletop environment for stompy's arm."""

from typing import Any

import numpy as np
import torch
import torch.random
from mani_skill.envs.tasks.tabletop.push_cube import PushCubeEnv, euler2quat
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import Array
from torch import Tensor

from stompy_live.agents.stompyarm.stompyarm import StompyArm
from stompy_live.utils.scene_builders.actor_builder import get_actor_builder
from stompy_live.utils.scene_builders.table_builder import StompyTableSceneBuilder


@register_env("PandaPushCube-v1", max_episode_steps=50)
class PandaPushCubeEnv(PushCubeEnv):
    SUPPORTED_ROBOTS = ["stompy_arm"]

    agent: StompyArm

    # Set some commonly used values
    goal_radius = 0.1
    cube_half_size = 0.02
    sphere_radius = 0.02

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        super().__init__(*args, robot_uids="stompy_arm", **kwargs)

    @property
    def _default_sensor_configs(self) -> list[CameraConfig]:
        # Registers one 128x128 camera looking at the robot, cube, and target
        # a smaller sized camera will be lower quality, but render faster
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
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
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100)

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

        self.sphere = actors.build_sphere(
            self.scene,
            radius=self.sphere_radius,
            color=np.array([12, 42, 160, 255]) / 255,
            name="ball",
            body_type="dynamic",
        )
        model_id = "024_bowl"
        self.builder = get_actor_builder(
            self.scene,
            id=f"ycb:{model_id}",
            scale=2,
        )
        self.bowl = self.builder.build(name="bowl")

        self.bowl2 = self.builder.build(name="bowl2")
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
            q = [1, 0, 0, 0]
            # obj_pose = Pose.create_from_pq(p=xyz, q=q)
            # self.obj.set_pose(obj_pose)

            target_region_xyz = xyz + torch.tensor([0.1 + self.goal_radius, -10000, 0])
            target_region_xyz[..., 2] = 1e-3
            self.goal_region.set_pose(
                Pose.create_from_pq(
                    p=target_region_xyz,
                    q=euler2quat(0, np.pi / 2, 0),
                )
            )

            sphere_xyz = torch.zeros((b, 3))
            sphere_xyz[..., :2] = self.sphere_radius
            # self.sphere.set_pose(sphere_pose)

            bowl_xyz = sphere_xyz + torch.tensor([0, 0.15, 0])
            bowl_pose = Pose.create_from_pq(p=bowl_xyz, q=q)
            self.bowl.set_pose(bowl_pose)

            bowl2_xyz = sphere_xyz + torch.tensor([0, -0.15, 0])
            bowl2_pose = Pose.create_from_pq(p=bowl2_xyz, q=q)
            self.bowl2.set_pose(bowl2_pose)

            sphere_xyz = sphere_xyz + torch.tensor([0, 0.15, 0])
            sphere_pose = Pose.create_from_pq(p=sphere_xyz, q=q)
            self.sphere.set_pose(sphere_pose)

            obj_xyz = sphere_xyz + torch.tensor([0, 0.05, 0])
            obj_pose = Pose.create_from_pq(p=obj_xyz, q=q)
            self.obj.set_pose(obj_pose)

    def evaluate(self) -> dict[str, Tensor]:
        cur_radius = torch.linalg.norm(self.obj.pose.p[..., :2] - self.goal_region.pose.p[..., :2], axis=1)
        is_obj_placed = cur_radius < self.goal_radius

        return {
            "success": is_obj_placed,
        }

    def compute_dense_reward(self, obs: Any, action: Array, info: dict) -> Tensor:  # noqa: ANN401
        # obj_to_goal_dist = torch.linalg.norm(self.obj.pose.p[..., :2] - self.goal_region.pose.p[..., :2], axis=1)
        # place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        # reward = place_reward
        # reward[info["success"]] = 3
        # return reward
        return torch.tensor(0.0)

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: dict) -> Tensor:  # noqa: ANN401
        # # This should be equal to compute_dense_reward / max possible reward
        # max_reward = 3.0
        # return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
        return torch.tensor(0.0)
