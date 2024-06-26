"""Defines the Stompy arm as an agent."""

from copy import deepcopy

import numpy as np
import sapien
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import PDJointVelControllerConfig
from mani_skill.agents.registration import register_agent

# from simgame.agents.controllers.keyboard import KeyboardControllerConfig
from stompy_live.utils.config import get_model_dir


def deepcopy_dict(configs: dict) -> dict:
    assert isinstance(configs, dict), type(configs)
    ret = {}
    for k, v in configs.items():
        if isinstance(v, dict):
            ret[k] = deepcopy_dict(v)
        else:
            ret[k] = deepcopy(v)
    return ret


@register_agent("stompy_arm")
class StompyArm(BaseAgent):
    uid = "stompy_arm"
    urdf_path = f"{get_model_dir()}/stompyarm/left_arm.urdf"

    urdf_config = {
        # "_materials": {
        #     "gripper": {
        #         "static_friction": 2.0,
        #         "dynamic_friction": 2.0,
        #         "restitution": 0.0,
        #     },
        # },
        # "link": {
        #     "link_right_arm_1_hand_1_gripper_1": {
        #         "material": "gripper",
        #         "patch_radius": 0.1,
        #         "min_patch_radius": 0.1,
        #     },
        #     "link_right_arm_1_hand_1_gripper_2": {
        #         "material": "gripper",
        #         "patch_radius": 0.1,
        #         "min_patch_radius": 0.1,
        #     },
        # },
    }

    startpos = [0.0, 0.0, 0.0]
    startorn = [0.0, 0.0, 0.0, 1.0]
    startrpy = [0.0, 0.0, 0.0]

    keyframes = {
        "rest": Keyframe(
            pose=sapien.Pose(p=startpos, q=startorn),
            qpos=np.array(
                [
                    [1.0, 1.0, 1.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0],
                ]
            ).flatten(),
        )
    }

    # fix_root_link = True
    # balance_passive_force = True
    # load_multiple_collisions = True

    arm_joint_names = [
        "joint_upper_left_arm_1_rmd_x8_90_mock_1_dof_x8",
        "joint_upper_left_arm_1_rmd_x8_90_mock_2_dof_x8",
        "joint_upper_left_arm_1_rmd_x4_24_mock_1_dof_x4",
        "joint_upper_left_arm_1_rmd_x4_24_mock_2_dof_x4",
        "joint_lower_arm_1_dof_1_rmd_x4_24_mock_2_dof_x4",
        "joint_lower_arm_1_dof_1_hand_1_rmd_x4_24_mock_1_dof_x4",
    ]

    ee_link_name = "link_lower_arm_1_dof_1_rmd_x4_24_mock_2_outer_rmd_x4_24_1"

    # arm_stiffness = 1e3
    # arm_damping = 1e2
    # arm_force_limit = 100

    arm_stiffness = 1
    arm_damping = 1
    arm_force_limit = 1

    gripper_stiffness = 1e3
    gripper_damping = 1e2
    gripper_force_limit = 100

    @property
    def _controller_configs(self) -> dict:
        return {
            "pd_joint_vel": PDJointVelControllerConfig(
                self.arm_joint_names,
                -1.0,
                1.0,
                self.arm_damping,  # this might need to be tuned separately
                self.arm_force_limit,
            ),
        }
