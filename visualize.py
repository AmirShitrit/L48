import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from agent import QLearningAgent
from env import Action, CellType, DroneGridEnv
from main import make_grid, train

_CELL_COLORS = {
    CellType.EMPTY: "white",
    CellType.WALL: "#555555",
    CellType.TRAP: "#FF6B6B",
    CellType.GOAL: "#51CF66",
}

_ARROW_DELTAS = {
    Action.UP: (0, 0.3),
    Action.DOWN: (0, -0.3),
    Action.LEFT: (-0.3, 0),
    Action.RIGHT: (0.3, 0),
}


def plot_learning_curve(win_history: list[int], report_every: int, path: Path) -> None:
    episodes = [i * report_every for i in range(1, len(win_history) + 1)]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, win_history, marker="o", linewidth=2, color="#4C9BE8")
    ax.axhline(report_every, color="gray", linestyle="--", linewidth=1, label="Perfect score")
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Wins per {report_every} episodes")
    ax.set_title("Learning Curve")
    ax.set_ylim(0, report_every + 5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _draw_cells(ax: plt.Axes, grid: np.ndarray) -> None:
    nrows, ncols = grid.shape
    for r in range(nrows):
        for c in range(ncols):
            cell = CellType(grid[r, c])
            rect = plt.Rectangle(
                (c, nrows - 1 - r), 1, 1,
                facecolor=_CELL_COLORS[cell], edgecolor="black", linewidth=0.5,
            )
            ax.add_patch(rect)


def _draw_policy_arrows(ax: plt.Axes, grid: np.ndarray, agent: QLearningAgent) -> None:
    nrows, ncols = grid.shape
    for r in range(nrows):
        for c in range(ncols):
            if CellType(grid[r, c]) != CellType.EMPTY:
                continue
            best_action = Action(agent.q_table[r, c].argmax())
            dx, dy = _ARROW_DELTAS[best_action]
            cx, cy = c + 0.5, nrows - 1 - r + 0.5
            ax.annotate(
                "", xy=(cx + dx, cy + dy), xytext=(cx, cy),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
            )


def plot_policy(grid: np.ndarray, agent: QLearningAgent, start: tuple[int, int], path: Path) -> None:
    nrows, ncols = grid.shape
    fig, ax = plt.subplots(figsize=(10, 10))

    _draw_cells(ax, grid)
    _draw_policy_arrows(ax, grid, agent)

    sr, sc = start
    ax.text(sc + 0.5, nrows - 1 - sr + 0.5, "S", ha="center", va="center", fontsize=14, fontweight="bold")

    legend = [mpatches.Patch(color=_CELL_COLORS[t], label=t.name.capitalize())
              for t in (CellType.WALL, CellType.TRAP, CellType.GOAL)]
    ax.legend(handles=legend, loc="upper right", fontsize=12)
    ax.set_xlim(0, ncols)
    ax.set_ylim(0, nrows)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Learned Policy", fontsize=16)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    assets = Path("assets")
    assets.mkdir(exist_ok=True)

    grid = make_grid()
    env = DroneGridEnv(grid, start=(0, 0))
    agent = QLearningAgent(n_rows=env.nrows, n_cols=env.ncols, n_actions=env.action_space.n)

    report_every = 100
    win_history = train(env, agent, report_every=report_every)

    plot_learning_curve(win_history, report_every, assets / "learning_curve.png")
    plot_policy(grid, agent, start=(0, 0), path=assets / "policy.png")
    print("Saved assets/learning_curve.png and assets/policy.png")


if __name__ == "__main__":
    main()
