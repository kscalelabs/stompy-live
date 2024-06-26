"""Defines a scene builder for putting Stompy on a table."""

import numpy as np
import sapien
import sapien.render
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from torch import Tensor
from transforms3d.euler import euler2quat


class StompyTableSceneBuilder(TableSceneBuilder):
    def initialize(self, env_idx: Tensor) -> None:
        # table_height = 0.9196429
        b = len(env_idx)
        pose = sapien.Pose(p=[-0.12, 0, -self.table_height], q=euler2quat(0, 0, np.pi / 2))
        self.table.set_pose(pose)

        if self.env.robot_uids == "stompy_arm":
            qpos = np.array([-0.129, 2.590, -1.786, 1.449, 0.0, 0.0, 0.0, 0.0])
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose(p=[-0.3, 0, 0], q=euler2quat(-3 * np.pi / 2, 0, -np.pi / 2)))

        else:
            raise KeyError(f"Unexpected robot UID: {self.env.robot_uids}")
