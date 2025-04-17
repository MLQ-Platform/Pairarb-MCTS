import numpy as np


class AllocationEnv:
    def __init__(self, states: np.ndarray, max_steps: int):
        """
        Initialize the environment.

        Args:
            states (np.ndarray): Binary signals of pairs.
            max_steps (int): Maximum steps allowed.
        """
        self.reset()
        self.states = states
        self.max_steps = max_steps

    def reset(self):
        """
        Reset cash and step counter.
        """
        self._cash = 1.0
        self._step = 0

    @property
    def is_terminal(self) -> bool:
        """
        Check if the environment is in a terminal state.

        Returns:
            bool: True if terminal, else False.
        """
        return self._step >= self.max_steps

    @property
    def now_state(self) -> np.ndarray:
        """
        Get the current state.

        Returns:
            np.ndarray: Current state signals.
        """
        return self.states[self._step]

    def execute(self, action: float) -> float:
        """
        Execute an action.

        Args:
            action (float): Exposure weight [0, 1].

        Returns:
            float: Reward from the action.
        """
        signal_cnt = np.sum(self.now_state == 1.0)

        if signal_cnt == 0:
            return 0.0

        exposure = min(self._cash, action)

        amount = exposure / signal_cnt

        rets = amount * self.now_state

        reward = np.sum(rets)

        self._cash = max(0, self._cash - exposure)

        self._step += 1

        return reward
