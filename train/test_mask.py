"""L1+L2 action-mask tests.

* C++ mask vs a Python BFS reference on random engine states.
* L1: wall/body moves are always masked out.
* Invariant: with no food and only mask-valid moves, a single snake can never
  die (the head always keeps a path to the tail, so it can never trap itself).
* L2: on a long, crowded snake the mask actually filters moves.
"""

import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import engine
from test_state_builder import can_reach_tail


def python_masks(game):
    W, H = game.gridWidth, game.gridHeight
    out = []
    for snake in game.snakes:
        mask = np.zeros(3, np.uint8)
        hx, hy = snake.body[0]
        for action in range(3):
            d = (snake.direction + action + 3) % 4
            dx = 1 if d == 0 else -1 if d == 2 else 0
            dy = 1 if d == 1 else -1 if d == 3 else 0
            nx, ny = hx + dx, hy + dy
            if nx < 0 or nx >= W or ny < 0 or ny >= H:
                continue
            blocked = False
            for other in game.snakes:
                for k in range(len(other.body) - 1):
                    if other.body[k] == (nx, ny):
                        blocked = True
                        break
                if blocked:
                    break
            if blocked:
                continue
            if not can_reach_tail(game, snake, nx, ny):
                continue
            mask[action] = 1
        out.append(mask)
    return np.stack(out) if out else np.zeros((0, 3), np.uint8)


def test_cpp_vs_python_random():
    rng = random.Random(7)
    games = engine.create_games(48, 32, 64)
    for _ in range(20):
        engine.tick(games)
        for g in games:
            g.remove_dead_snakes()
            # respawn so the population stays representative
            if len(g.snakes) < 5:
                g.add_delay_snake(1)
        _, masks_cpp = engine.build_states_and_masks(games, 23)
        masks_py = np.concatenate([python_masks(g) for g in games])
        assert masks_cpp.shape == masks_py.shape
        assert np.array_equal(masks_cpp, masks_py), (
            "C++ and Python masks differ on a random engine state"
        )
    print("C++ mask == Python reference on random states: OK")


def test_l1_wall_and_body():
    g = engine.Game(8, 8)
    s = engine.Snake()
    # head at (0,4) facing -x straight into the wall; tail behind it.
    s.body = [(0, 4), (1, 4), (2, 4)]
    s.direction = 2
    s.alive = True
    s.foodCount = 0
    g.snakes = [s]
    g.foods = []
    masks = python_masks(g)
    assert masks[0][1] == 0, "wall move must be masked"
    assert masks[0][0] == 1 and masks[0][2] == 1
    # also via the engine
    _, masks_cpp = engine.build_states_and_masks([g], 23)
    assert np.array_equal(masks_cpp[0], masks[0])
    print("L1 wall/body masking: OK")


def _single_snake_game(body, direction):
    g = engine.Game(32, 64)
    s = engine.Snake()
    s.body = list(body)
    s.direction = direction
    s.alive = True
    s.foodCount = 0
    g.snakes = [s]
    g.foods = []
    return g


def test_invariant_no_food_never_dies():
    rng = random.Random(11)
    trials = 24
    ticks = 4000
    for trial in range(trials):
        g = _single_snake_game(
            [(16, 32), (15, 32), (14, 32)], 0
        )
        for tick in range(ticks):
            _, masks = engine.build_states_and_masks([g], 23)
            valid = [a for a in range(3) if masks[0, a]]
            assert valid, "mask-valid set empty before death (invariant broken)"
            g.snakes[0].setDirection(rng.choice(valid) - 1)
            deaths = engine.tick([g])
            if deaths:
                raise AssertionError(
                    f"trial {trial}: snake died at tick {tick} under masked policy"
                )
    print(f"invariant: {trials} x {ticks} ticks, single snake never died: OK")


def test_l2_filters_on_crowded_snake():
    rng = random.Random(23)
    g = _single_snake_game([(16, 32), (15, 32), (14, 32)], 0)
    total_masked = 0
    total_actions = 0
    seen_filtered = False
    for tick in range(12000):
        if len(g.foods) < 2:
            for _ in range(2 - len(g.foods)):
                while True:
                    x, y = rng.randrange(32), rng.randrange(64)
                    if not any(seg == (x, y) for seg in g.snakes[0].body):
                        break
                g.foods.append((x, y))
        _, masks = engine.build_states_and_masks([g], 23)
        total_actions += 3
        total_masked += int(np.count_nonzero(masks[0] == 0))
        if np.any(masks[0] == 0):
            seen_filtered = True
        valid = [a for a in range(3) if masks[0, a]]
        if not valid:
            break  # snake filled the board; that is expected
        g.snakes[0].setDirection(rng.choice(valid) - 1)
        deaths = engine.tick([g])
        if deaths:
            break
    assert seen_filtered, "L2/L1 never filtered a move on a crowded snake"
    frac = total_masked / max(total_actions, 1)
    print(f"crowded snake: {frac * 100:.1f}% of actions masked, filter active: OK")


if __name__ == "__main__":
    test_cpp_vs_python_random()
    test_l1_wall_and_body()
    test_invariant_no_food_never_dies()
    test_l2_filters_on_crowded_snake()
    print("all mask tests passed")
