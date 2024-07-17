"""Defines class for the stompy agent."""

from typing import Any, Dict

import numpy as np
import sapien
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import PDJointPosControllerConfig
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from transforms3d import euler

from stompy_live.utils.config import get_model_dir


@register_agent(
    "stompy_latest"
)  # uncomment this if you want to register the agent so you can instantiate it by ID when creating environments
class Stompy(BaseAgent):
    uid = "stompy_latest"
    urdf_path = f"{get_model_dir()}/7DOF_NEWEST/robot_7dof_arm_merged_simplified.urdf"
    urdf_config = dict(
        _materials=dict(gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)),
        link=dict(
            link_left_arm_2_hand_1_gripper_1=dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
            link_left_arm_2_hand_1_gripper_2=dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
            link_right_arm_1_hand_1_gripper_1=dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
            link_right_arm_1_hand_1_gripper_2=dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
        ),
    )
    fix_root_link = False
    disable_self_collisions = True

    keyframes = dict(
        standing=Keyframe(
            pose=sapien.Pose(p=[0, 0, 0.965]),
            qpos=np.array(
                [
                    [
                        1.5,
                        -1.5,
                        -0.942,
                        0,
                        1.5,
                        -1.5,
                        -0.25,
                        0.25,
                        0.5,
                        0.65,
                        -0.65,
                        -0.5,
                        -0.5,
                        0,
                        0,
                        -0.5,
                        0.5,
                        0.78,
                        -0.78,
                        0.25,
                        -0.25,
                        0,
                        -0.2,
                        0,
                        0.2,
                        0,
                        0,
                        -0.2,
                        0.2,
                        0.2,
                        -2.2,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ]
                ]
            ),
        )
    )

    @property
    def _controller_configs(
        self,
    ) -> Dict[str, Any]:
        return dict(
            pd_joint_pos=dict(
                body=PDJointPosControllerConfig(
                    [x.name for x in self.robot.active_joints],
                    lower=None,
                    upper=None,
                    stiffness=100,
                    damping=10,
                    normalize_action=False,
                ),
                balance_passive_force=False,
            ),
            pd_joint_delta_pos=dict(
                body=PDJointPosControllerConfig(
                    [x.name for x in self.robot.active_joints],
                    lower=-0.1,
                    upper=0.1,
                    stiffness=20,
                    damping=5,
                    normalize_action=True,
                    use_delta=True,
                ),
                balance_passive_force=False,
            ),
        )

    @property
    def _sensor_configs(self) -> None:
        return [
            CameraConfig(
                uid="head_camera",
                pose=sapien.Pose(p=[0.12, 0, 0.02], q=euler.euler2quat(-np.pi / 2, 0, 0)),
                width=128,
                height=128,
                fov=1.57,
                near=0.01,
                far=100,
            )
        ]


# TODO (add a simplified stompy)
