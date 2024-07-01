"""Defines the Stompy torso as an agent."""

from copy import deepcopy

import numpy as np
import sapien
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import PDJointPosControllerConfig, PDJointVelControllerConfig
from mani_skill.agents.registration import register_agent
from mani_skill.utils import sapien_utils

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


@register_agent("stompy_torso")
class StompyTorso(BaseAgent):
    uid = "stompy_torso"
    urdf_path = f"{get_model_dir()}/torso_merged_simplified/upper_limb_assembly_5_dof_merged_simplified.urdf"

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

    # print links in robot

    startpos = [0.0, 0.0, 0.0]
    startorn = [0.0, 0.0, 0.0, 1.0]
    startrpy = [0.0, 0.0, 0.0]

    keyframes = {
        "rest": Keyframe(
            pose=sapien.Pose(p=startpos, q=startorn),
            qpos=np.array(
                [1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            ).flatten(),
            # qpos=np.array([1.0, 1.0, 1.0]),
        )
    }

    # fix_root_link = True
    # balance_passive_force = True
    # load_multiple_collisions = True

    arm_joint_names = []

    # ee_link_name = "fused_link_full_arm_5_dof_1_lower_arm_1_dof_1_rmd_x4_24_mock_2_outer_rmd_x4_24_1"
    ee_link_name = "fused_component_full_arm_5_dof_2_lower_arm_1_dof_1_hand_1_slide_2"
    # tcp_link_name = "fused_link_full_arm_5_dof_1_lower_arm_1_dof_1_hand_1_spur_gear_26_teeth_1"

    arm_stiffness = 1e3
    arm_damping = 10
    arm_force_limit = 100

    gripper_stiffness = 1e3
    gripper_damping = 1e2
    gripper_force_limit = 100

    @property
    def _controller_configs(self) -> dict:
        print("joint names", [j.name for j in self.robot.active_joints])
        return {
            "pd_joint_vel": PDJointVelControllerConfig(
                [j.name for j in self.robot.active_joints],
                -1.0,
                1.0,
                self.arm_damping,  # this might need to be tuned separately
                self.arm_force_limit,
            ),
            "pd_joint_delta_pos": PDJointPosControllerConfig(
                [j.name for j in self.robot.active_joints],
                lower=-0.1,
                upper=0.1,
                stiffness=self.arm_stiffness,
                damping=self.arm_damping,
                force_limit=self.arm_force_limit,
                use_delta=True,
            ),
        }

    def _after_init(self) -> None:
        self.tcp = sapien_utils.get_obj_by_name(self.robot.get_links(), self.ee_link_name)
        self.ee = sapien_utils.get_obj_by_name(self.robot.get_links(), self.ee_link_name)
