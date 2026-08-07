"""Correctness check + micro benchmark for the C++ observation builder."""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import DQNAgent
from game import Game as PyGame
from snake import Snake as PySnake
import engine


def python_reference(agent, games):
    states, masks = [], []
    for game in games:
        game_state = agent.getGameState(
            game.gridWidth, game.gridHeight, game.snakes, game.foodManager.foods
        )
        for snake in game.snakes:
            states.append(
                agent.getState(
                    snake, game.gridWidth, game.gridHeight, game_state, game.foodManager.foods
                )
            )

            valid = []
            for action in range(3):
                direction = (snake.direction + action + 3) % 4
                x, y = (
                    snake.body[0][0] + [1, 0, -1, 0][direction],
                    snake.body[0][1] + [0, 1, 0, -1][direction],
                )
                if (
                    x < 0
                    or x >= game.gridWidth
                    or y < 0
                    or y >= game.gridHeight
                    or any(
                        any((tx == x and ty == y) for tx, ty in other.body[:-1])
                        for other in game.snakes
                    )
                ):
                    continue
                if not can_reach_tail(game, snake, x, y):
                    continue
                valid.append(action)

            mask = np.zeros(3, np.uint8)
            mask[valid] = 1
            masks.append(mask)

    return np.stack([s.numpy() for s in states]), np.stack(masks)


def can_reach_tail(game, snake, headX, headY):
    """L2 invariant: after moving the head to (headX, headY), can it still
    reach the post-move tail (old second-to-last body segment)? The old tail
    cell vacates; a growth tick only duplicates the new tail in place."""
    W, H = game.gridWidth, game.gridHeight
    goal = snake.body[len(snake.body) - 2]
    blocked = np.zeros(W * H, bool)

    for k in range(len(snake.body) - 2):
        x, y = snake.body[k]
        if 0 <= x < W and 0 <= y < H:
            blocked[y * W + x] = True
    for other in game.snakes:
        if other is snake:
            continue
        for k in range(len(other.body) - 1):
            x, y = other.body[k]
            if 0 <= x < W and 0 <= y < H:
                blocked[y * W + x] = True

    gx, gy = goal
    if not (0 <= gx < W and 0 <= gy < H):
        return False
    blocked[gy * W + gx] = False
    if (headX, headY) == (gx, gy):
        return True

    queue = [headY * W + headX]
    blocked[headY * W + headX] = True
    head = 0
    while head < len(queue):
        cur = queue[head]
        head += 1
        cx, cy = cur % W, cur // W
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = cx + dx, cy + dy
            if nx < 0 or nx >= W or ny < 0 or ny >= H:
                continue
            idx = ny * W + nx
            if blocked[idx]:
                continue
            if (nx, ny) == (gx, gy):
                return True
            blocked[idx] = True
            queue.append(idx)
    return False


def make_pair(layout):
    """Python game + engine game with an identical state layout."""
    pg = PyGame(32, 64)
    pg.snakes = []
    pg.foodManager.foods = []
    es_list = []
    for body, direction in layout["snakes"]:
        ps = PySnake(pg)
        ps.body = list(body)
        ps.direction = direction
        ps.survivalCount = 0
        ps.foodCount = 0
        ps.growCount = 0
        ps.ateCount = 0
        ps.alive = True
        pg.snakes.append(ps)

        es = engine.Snake()
        es.body = list(body)
        es.direction = direction
        es_list.append(es)
    pg.foodManager.foods = list(layout["foods"])
    eg = engine.Game(32, 64)
    eg.foods = list(layout["foods"])
    eg.snakes = es_list
    return pg, eg


def check(agent, pg, eg, label):
    states_cpp, masks_cpp = engine.build_states_and_masks([eg], agent.viewSize)
    states_py, masks_py = python_reference(agent, [pg])

    assert states_cpp.shape == states_py.shape, (states_cpp.shape, states_py.shape)
    diff = float(np.abs(states_cpp - states_py).max())
    ok_masks = bool(np.array_equal(masks_cpp, masks_py))
    print(f"{label}: states max abs diff = {diff:.3e}, masks equal = {ok_masks}")
    assert diff < 1e-5, f"state mismatch: {diff}"
    assert ok_masks, "mask mismatch"


def main():
    agent = DQNAgent()

    # Varied layouts, incl. snakes near walls and long bodies.
    layouts = [
        {"snakes": [([(10, 10), (9, 10), (8, 10)], 0),
                    ([(20, 20), (20, 19), (20, 18)], 1),
                    ([(0, 30), (0, 29), (0, 28)], 2)],  # hugging left wall
         "foods": [(12, 10), (5, 5), (30, 40), (2, 2), (31, 63)]},
        {"snakes": [([(1, 1)] * 40, 3), ([(31, 62)] * 60, 0)],  # long bodies
         "foods": [(0, 0), (31, 0), (0, 63), (15, 30), (16, 30), (17, 30)]},
    ]
    for i, layout in enumerate(layouts):
        pg, eg = make_pair(layout)
        check(agent, pg, eg, f"layout {i}")

    # Random layouts via the engine, cross-checked against the Python reference
    # on an identical snapshot taken from the engine games.
    big = engine.create_games(128, 32, 64)
    engine.tick(big)
    for eg in big:
        eg.remove_dead_snakes()
    py_big = []
    es_big = []
    for eg in big:
        pg = PyGame(32, 64)
        pg.snakes = []
        pg.foodManager.foods = []
        for s in eg.snakes:
            ps = PySnake(pg)
            ps.body = list(s.body)
            ps.direction = s.direction
            ps.alive = s.alive
            ps.ate = s.ate
            ps.survivalCount = s.survivalCount
            ps.foodCount = s.foodCount
            ps.growCount = s.growCount
            ps.ateCount = s.ateCount
            pg.snakes.append(ps)
        pg.foodManager.foods = list(eg.foods)
        py_big.append(pg)
        es_big.append(eg)

    states_cpp, masks_cpp = engine.build_states_and_masks(es_big, agent.viewSize)
    states_py, masks_py = python_reference(agent, py_big)
    diff = float(np.abs(states_cpp - states_py).max())
    print(f"128 games snapshot: states max abs diff = {diff:.3e}, masks equal = "
          f"{bool(np.array_equal(masks_cpp, masks_py))}")
    assert diff < 1e-5 and np.array_equal(masks_cpp, masks_py)

    # Micro benchmark.
    n_calls = 30
    t0 = time.perf_counter()
    for _ in range(n_calls):
        engine.build_states_and_masks(big, agent.viewSize)
    cpp_time = (time.perf_counter() - t0) / n_calls

    t0 = time.perf_counter()
    for _ in range(2):
        python_reference(agent, py_big)
    py_time = (time.perf_counter() - t0) / 2

    n_snakes = sum(len(g.snakes) for g in big)
    print(f"snakes: {n_snakes}")
    print(f"cpp   build: {cpp_time * 1000:.2f} ms/call")
    print(f"python ref: {py_time * 1000:.2f} ms/call")
    print(f"speedup: {py_time / cpp_time:.1f}x")


if __name__ == "__main__":
    main()
