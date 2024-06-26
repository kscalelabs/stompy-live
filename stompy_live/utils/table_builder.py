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
            qpos = np.array(
                [
                    0.0,
                    np.pi / 8,
                    0,
                    -np.pi * 5 / 8,
                    0,
                    np.pi * 3 / 4,
                    np.pi / 4,
                    0.04,
                    0.04,
                ]
            )
            qpos = self.env._episode_rng.normal(0, self.robot_init_qpos_noise, (b, len(qpos))) + qpos
            qpos[:, -2:] = 0.04
            self.env.agent.robot.set_pose(sapien.Pose(p=[-0.3, 0, 0], q=euler2quat(-3 * np.pi / 2, 0, -np.pi / 2)))

        else:
            raise KeyError(f"Unexpected robot UID: {self.env.robot_uids}")
