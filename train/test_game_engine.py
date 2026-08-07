"""Equivalence + invariant tests for the C++ game engine vs Python reference."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import engine
from game import Game as PyGame
from snake import Snake as PySnake, deadInfo


class DummyAgent:
    steps = 3
    gamma = 0.95

    class Memory:
        alpha = 0.6

    memory = Memory()


def make_pair(layout):
    pg = PyGame(32, 64)
    pg.snakes = []
    pg.delaySnakes = []
    pg.foodManager.delayFoods = []
    pg.foodManager.deltaTime = 0
    pg.foodManager.foods = []

    eg = engine.Game(32, 64)
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
        ps.ate = False
        pg.snakes.append(ps)

        es = engine.Snake()
        es.body = list(body)
        es.direction = direction
        es_list.append(es)

    pg.foodManager.foods = list(layout["foods"])
    eg.foods = list(layout["foods"])
    eg.snakes = es_list
    return pg, eg


def snapshot(pg, eg):
    def py_snap():
        return [
            (list(s.body), s.direction, s.alive, s.survivalCount, s.foodCount,
             s.growCount, s.ateCount, s.ate)
            for s in pg.snakes
        ]

    def eg_snap():
        return [
            (list(s.body), s.direction, s.alive, s.survivalCount, s.foodCount,
             s.growCount, s.ateCount, s.ate)
            for s in eg.snakes
        ]

    return py_snap(), eg_snap(), list(pg.foodManager.foods), list(eg.foods)


def post_tick_python(pg):
    pg.postTick(DummyAgent())


def post_tick_engine(eg):
    from sim import rememberStageB
    for snake in eg.snakes:
        rememberStageB(snake, DummyAgent())
        if not snake.alive:
            for j in range(0, len(snake.body), 2):
                eg.add_delay_food(snake.body[j][0], snake.body[j][1], 6)
            eg.add_delay_snake(20)
    eg.remove_dead_snakes()


def check_deterministic_ticks():
    # Two snakes that move without hitting walls, one food in the path of the
    # first snake so growth is exercised. No food spawns within 3 ticks.
    layout = {
        "snakes": [
            ([(10, 10), (9, 10), (8, 10)], 0),  # moving +x
            ([(20, 20), (20, 19), (20, 18)], 1),  # moving +y
        ],
        "foods": [(12, 10), (30, 30), (31, 31), (5, 5), (6, 6), (7, 7)],
    }
    pg, eg = make_pair(layout)

    for tick in range(3):
        pg.tick()
        deaths_py = []
        deadInfo.clear()
        # Python appends to deadInfo during tick; keep a copy.
        deaths_py.extend(deadInfo)

        eg_deaths = list(engine.tick([eg]))

        post_tick_python(pg)
        post_tick_engine(eg)

        py_s, eg_s, py_f, eg_f = snapshot(pg, eg)
        assert py_s == eg_s, f"tick {tick} snake mismatch:\n{py_s}\n{eg_s}"
        assert py_f == eg_f, f"tick {tick} foods mismatch: {py_f} vs {eg_f}"
        assert deaths_py == list(eg_deaths), (
            f"tick {tick} deaths mismatch: {deaths_py} vs {eg_deaths}"
        )
    print("deterministic ticks: OK")


def check_wall_death_and_respawn():
    layout = {
        "snakes": [([(30, 5), (29, 5), (28, 5)], 0)],  # dies at x=32
        "foods": [],
    }
    pg, eg = make_pair(layout)
    for _ in range(2):  # head 30 -> 31 -> 32 (wall)
        deadInfo.clear()
        pg.tick()
        deaths_py = list(deadInfo)
        eg_deaths = list(engine.tick([eg]))
        assert deaths_py == eg_deaths, f"death records: {deaths_py} vs {eg_deaths}"
        post_tick_python(pg)
        post_tick_engine(eg)

    assert len(pg.snakes) == 0 and len(eg.snakes) == 0, "dead snake not removed"
    assert len(pg.delaySnakes) == 1 and len(eg.delaySnakes) == 1
    assert len(pg.foodManager.delayFoods) == len(eg.delayFoods)
    assert len(eg.delayFoods) == (len(layout["snakes"][0][0]) + 1) // 2

    # Respawn after the delay elapses; stop as soon as the snake is back so no
    # further random state diverges the comparison.
    for _ in range(21):
        pg.tick()
        engine.tick([eg])
        post_tick_python(pg)
        post_tick_engine(eg)
        assert len(eg.snakes) <= 1 and len(pg.snakes) <= 1
        if len(pg.snakes) == 1:
            break
    assert len(pg.snakes) == 1 and len(eg.snakes) == 1, "respawn failed"
    assert 3 <= len(eg.snakes[0].body) <= 32
    hx, hy = eg.snakes[0].body[0]
    assert 0 <= hx < 32 and 0 <= hy < 64, "respawned outside grid"
    print("wall death + respawn: OK")


def check_delay_food_placement():
    layout = {"snakes": [([(10, 10), (9, 10), (8, 10)], 0)], "foods": []}
    pg, eg = make_pair(layout)
    # Delayed food at a fixed position (deterministic, no RNG involved).
    pg.foodManager.delayFoods.append([1, (20, 20)])
    eg.add_delay_food(20, 20, 1)
    pg.tick()
    engine.tick([eg])
    assert (20, 20) in pg.foodManager.foods and (20, 20) in list(eg.foods)
    print("delay food placement: OK")


def check_invariants_over_random_games():
    games = engine.create_games(128, 32, 64)
    for tick in range(200):
        engine.tick(games)
        for g in games:
            g.remove_dead_snakes()
            assert len(g.snakes) <= 5
            assert len(g.foods) >= 3 and len(g.foods) <= g.gridWidth * g.gridHeight // 50
            for s in g.snakes:
                assert s.alive
                assert 3 <= len(s.body)
                hx, hy = s.body[0]
                assert 0 <= hx < g.gridWidth and 0 <= hy < g.gridHeight
    print("200 random ticks over 128 games: OK")


if __name__ == "__main__":
    check_deterministic_ticks()
    check_wall_death_and_respawn()
    check_delay_food_placement()
    check_invariants_over_random_games()
