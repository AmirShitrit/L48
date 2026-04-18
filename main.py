import numpy as np

from agent import QLearningAgent
from env import CellType, DroneGridEnv


def make_grid() -> np.ndarray:
    grid = np.full((10, 10), CellType.EMPTY, dtype=np.int32)
    grid[9, 9] = CellType.GOAL
    for col in range(1, 5):
        grid[2, col] = CellType.WALL
    for col in range(5, 9):
        grid[5, col] = CellType.WALL
    for col in range(1, 5):
        grid[7, col] = CellType.WALL
    for row, col in [(1, 6), (3, 8), (4, 2), (6, 1), (9, 4)]:
        grid[row, col] = CellType.TRAP
    return grid


def _to_state(obs: np.ndarray) -> tuple[int, int]:
    return int(obs[0]), int(obs[1])


def run_episode(env: DroneGridEnv, agent: QLearningAgent, max_steps: int = 200) -> bool:
    obs, _ = env.reset()
    state = _to_state(obs)

    for _ in range(max_steps):
        action = agent.choose_action(state)
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_state = _to_state(next_obs)

        agent.update(state, action, reward, next_state, terminated)
        state = next_state

        if terminated:
            return info["cell"] == CellType.GOAL
        if truncated:
            return False

    return False


def train(env: DroneGridEnv, agent: QLearningAgent, n_episodes: int = 1500, report_every: int = 100) -> None:
    wins = 0
    for episode in range(n_episodes):
        if run_episode(env, agent):
            wins += 1
            agent.decay_epsilon()
        if (episode + 1) % report_every == 0:
            start = episode + 2 - report_every
            print(f"Episodes {start}–{episode + 1}: {wins}/{report_every} wins")
            wins = 0


def main() -> None:
    env = DroneGridEnv(make_grid(), start=(0, 0))
    print(env.render())
    agent = QLearningAgent(n_rows=env.nrows, n_cols=env.ncols, n_actions=env.action_space.n)
    train(env, agent)
    agent.save("qtable.json")
    print("Q-table saved to qtable.json")


if __name__ == "__main__":
    main()
