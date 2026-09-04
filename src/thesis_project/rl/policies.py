from typing import Any, Protocol

import numpy as np


class Predictor(Protocol):
    def predict(
        self,
        observation: Any,
        *,
        deterministic: bool = True,
        action_masks: np.ndarray | None = None,
    ) -> tuple[Any, Any]: ...


class FixedActionPolicy:
    def __init__(self, action: int, fallback_action: int = 1):
        self.action = action
        self.fallback_action = fallback_action

    def predict(
        self,
        observation: Any,
        *,
        deterministic: bool = True,
        action_masks: np.ndarray | None = None,
    ) -> tuple[int, None]:
        if action_masks is None:
            raise ValueError("FixedActionPolicy requires action_masks.")

        if action_masks[self.action]:
            return self.action, None

        if not action_masks[self.fallback_action]:
            raise ValueError(
                f"Neither action {self.action} nor fallback action "
                f"{self.fallback_action} is valid."
            )

        return self.fallback_action, None


class FixedLongPolicy(FixedActionPolicy):
    def __init__(self):
        super().__init__(action=2)


class FixedFlatPolicy(FixedActionPolicy):
    def __init__(self):
        super().__init__(action=1)


class FixedShortPolicy(FixedActionPolicy):
    def __init__(self):
        super().__init__(action=0)


class RandomPolicy:
    def predict(
        self,
        observation: Any,
        *,
        deterministic: bool = True,
        action_masks: np.ndarray | None = None,
    ) -> tuple[int, None]:
        if action_masks is None:
            raise ValueError("RandomPolicy requires action_masks.")

        valid_actions = np.flatnonzero(action_masks)
        return int(np.random.choice(valid_actions)), None
