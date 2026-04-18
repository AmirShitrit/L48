from enum import IntEnum

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CellType(IntEnum):
    EMPTY = 0
    WALL = 1
    TRAP = 2
    GOAL = 3


class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


_REWARDS = {CellType.EMPTY: -1, CellType.TRAP: -10, CellType.GOAL: 10}
_ICONS = {CellType.EMPTY: "  ", CellType.WALL: "🧱", CellType.TRAP: "🧨", CellType.GOAL: "🎯"}
_DELTAS = {Action.UP: (-1, 0), Action.DOWN: (1, 0), Action.LEFT: (0, -1), Action.RIGHT: (0, 1)}


class DroneGridEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, grid: np.ndarray, start: tuple[int, int], render_mode: str | None = None):
        super().__init__()
        self._grid = grid
        self._start = start
        self.render_mode = render_mode
        self.nrows, self.ncols = grid.shape
        self.observation_space = spaces.MultiDiscrete([self.nrows, self.ncols])
        self.action_space = spaces.Discrete(len(Action))
        self._pos = np.array(start)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._pos = np.array(self._start)
        return self._pos.copy(), {}

    def step(self, action: int):
        dr, dc = _DELTAS[Action(action)]
        new_row, new_col = self._pos[0] + dr, self._pos[1] + dc

        if (0 <= new_row < self.nrows
                and 0 <= new_col < self.ncols
                and self._grid[new_row, new_col] != CellType.WALL):
            self._pos = np.array([new_row, new_col])

        cell = CellType(self._grid[self._pos[0], self._pos[1]])
        terminated = cell in (CellType.GOAL, CellType.TRAP)
        return self._pos.copy(), _REWARDS[cell], terminated, False, {"cell": cell}

    def _render_row(self, r: int) -> str:
        pos = (int(self._pos[0]), int(self._pos[1]))
        cells = ("🛩️" if (r, c) == pos else _ICONS[CellType(self._grid[r, c])] for c in range(self.ncols))
        return "|" + "".join(cells) + "|"

    def render(self) -> str:
        border = "+" + "──" * self.ncols + "+"
        rows = [self._render_row(r) for r in range(self.nrows)]
        return "\n".join([border, *rows, border])
