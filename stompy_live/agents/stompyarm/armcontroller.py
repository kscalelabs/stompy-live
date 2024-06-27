"""Custom controller class for stompy's arm."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from gymnasium import spaces
from mani_skill.agents.controllers.base_controller import BaseController, ControllerConfig


class StompyController(BaseController):
    config: "StompyControllerConfig"

    def _initialize_action_space(self) -> None:
        n = len(self.joints)
        low = np.float32(np.broadcast_to(self.config.lower, n))
        high = np.float32(np.broadcast_to(self.config.upper, n))
        self.single_action_space = spaces.Box(low, high, dtype=np.float32)

    def set_drive_property(self) -> None:
        pass

    def set_action(self, action: np.ndarray) -> None:
        # action = self._preprocess_action(action)
        # self.articulation.set_joint_drive_velocity_targets(action, self.joints, self.active_joint_indices)
        pass


@dataclass
class StompyControllerConfig(ControllerConfig):
    controller_cls = StompyController
    lower: float | Sequence[float]
    upper: float | Sequence[float]
