"""Fixed masked-greedy evaluation of candidate checkpoints.

Runs a fixed number of ticks over fresh arenas and reports average survival /
food eaten per snake, so checkpoints can be compared apples-to-apples.
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import DQNAgent
import engine


def evaluate(path, ticks):
    agent = DQNAgent()
    agent.model.load_weights(path)
    agent.targetModel.load_weights(path)
    agent.epsilon = 0.0  # greedy
    games = engine.create_games(64, 32, 64)

    for g in games:
        g.remove_dead_snakes()
        while len(g.snakes) < 5:
            g.add_delay_snake(1)

    survivals = []
    ates = []
    alive_at_end = 0

    for tick in range(ticks):
        states, masks = engine.build_states_and_masks(games, agent.viewSize)
        values = agent.predict(states).numpy()

        masked = np.where(masks == 1, values, -1e9)
        actions = np.argmax(masked, -1)
        all_invalid = masks.sum(axis=1) == 0
        if all_invalid.any():
            actions[all_invalid] = np.argmax(values[all_invalid], -1)

        idx = 0
        for game in games:
            for snake in game.snakes:
                snake.setDirection(int(actions[idx]) - 1)
                idx += 1

        for death in engine.tick(games):
            survivals.append(death[0])
            ates.append(death[1])

        for game in games:
            game.remove_dead_snakes()
            if len(game.snakes) < 5:
                game.add_delay_snake(1)

    for game in games:
        for snake in game.snakes:
            alive_at_end += 1
            survivals.append(snake.survivalCount)
            ates.append(snake.ateCount)

    out = {
        "model": os.path.basename(path),
        "ticks": ticks,
        "deaths": len(survivals),
        "avg_survival": float(np.mean(survivals)),
        "avg_ate": float(np.mean(ates)),
        "max_survival": int(np.max(survivals)),
        "max_ate": int(np.max(ates)),
        "alive_at_end": alive_at_end,
    }
    print(json_dumps(out), flush=True)
    return out


def json_dumps(d):
    import json
    return json.dumps(d)


def main():
    paths = sys.argv[1:]
    ticks = 3000
    for p in paths:
        t0 = time.perf_counter()
        evaluate(p, ticks)
        print(f"  ({time.perf_counter() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
