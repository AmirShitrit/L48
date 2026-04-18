import json
from pathlib import Path

import numpy as np


class QLearningAgent:
    def __init__(
        self,
        n_rows: int,
        n_cols: int,
        n_actions: int,
        learning_rate: float = 0.1,
        discount: float = 0.9,
        initial_epsilon: float = 0.3,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
    ):
        self.q_table = np.zeros((n_rows, n_cols, n_actions))
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = initial_epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self._rng = np.random.default_rng()

    def _greedy_action(self, state: tuple[int, int]) -> int:
        q_vals = self.q_table[state[0], state[1]]
        best = np.flatnonzero(q_vals == q_vals.max())
        return int(self._rng.choice(best))

    def choose_action(self, state: tuple[int, int]) -> int:
        if self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.q_table.shape[2]))
        return self._greedy_action(state)

    def update(self, state: tuple[int, int], action: int, reward: float,
               next_state: tuple[int, int], terminated: bool) -> None:
        next_max = 0.0 if terminated else self.q_table[next_state[0], next_state[1]].max()
        td_error = reward + self.gamma * next_max - self.q_table[state[0], state[1], action]
        self.q_table[state[0], state[1], action] += self.lr * td_error

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str | Path) -> None:
        data = {
            "q_table": self.q_table.tolist(),
            "learning_rate": self.lr,
            "discount": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "QLearningAgent":
        data = json.loads(Path(path).read_text())
        q = np.array(data["q_table"])
        n_rows, n_cols, n_actions = q.shape
        agent = cls(n_rows, n_cols, n_actions,
                    learning_rate=data["learning_rate"],
                    discount=data["discount"],
                    initial_epsilon=data["epsilon"],
                    epsilon_min=data.get("epsilon_min", 0.01),
                    epsilon_decay=data.get("epsilon_decay", 0.995))
        agent.q_table = q
        return agent
