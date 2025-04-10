import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import random

from snake import Snake, deadInfo
from food import FoodManager
from agent import DQNAgent
import config as cfg

class TimerManager:

    def __init__(self):
        self.timers = []

    def setTimer(self, callback, delay):
        self.timers.append([callback, delay])

    def gameplayTick(self):
        for i in range(len(self.timers)):
            self.timers[i][1] -= 1

            if self.timers[i][1] <= 0:
                self.timers[i][0]()

        self.timers = list(filter(lambda t: t[1] > 0, self.timers))


writer = tf.summary.create_file_writer('log/')
timerManager = TimerManager()
cfg.foodManager = FoodManager()
cfg.agent = DQNAgent()
maxGrows = 1
iter = 0

for _ in range(5):
    cfg.foodManager.generateFood()

for _ in range(5):
    cfg.snakes.append(Snake(maxGrows))


cfg.agent.model.summary()

while True:
    if len(cfg.snakes):
        states = [cfg.agent.getState(snake) for snake in cfg.snakes]
        values = cfg.agent.forward(cfg.agent.model, tf.stack(states)).numpy()

        for i, snake in enumerate(cfg.snakes):
            actions = []

            for action in range(4):
                if (snake.direction - action + 4) % 4 == 2:
                    continue

                x, y = snake.body[0]
                x, y = x + [1, 0, -1, 0][action], y + [0, 1, 0, -1][action]

                if x < 0 or x >= cfg.gridWidth or y < 0 or y >= cfg.gridHeight:
                    continue

                if any(any((tx == x and ty == y) for tx, ty in snake.body[:-1]) for snake in cfg.snakes):
                    continue

                actions.append(action)

            actions = actions if len(actions) else [0, 1, 2, 3]

            for action in range(4):
                if action not in actions:
                    values[i, action] = -999

            action = random.choice(actions) if random.random() < cfg.agent.epsilon else int(np.argmax(values[i]))
            snake.rememberStageA(states[i], action, float(values[i, action]))
            snake.setDirection(action)

        for snake in cfg.snakes: snake.move()
        for snake in cfg.snakes: snake.checkCollision()
        for snake in cfg.snakes: snake.checkFood()

    timerManager.gameplayTick()
    cfg.foodManager.gameplayTick()

    for snake in cfg.snakes:
        snake.rememberStageB()

        if not snake.alive:
            for i in range(len(snake.body)):
                if i % 2 == 0:
                    def func(x, y):
                        return lambda: cfg.foodManager.generateFood(x, y)

                    timerManager.setTimer(func(snake.body[i][0], snake.body[i][1]), 6)

            timerManager.setTimer(lambda: cfg.snakes.append(Snake(maxGrows)), 20)

    cfg.snakes = list(filter(lambda snake: snake.alive, cfg.snakes))

    if len(cfg.agent.memory.memories) >= 1024 and cfg.agent.memory.newSize >= 512:
        iter += 1
        cfg.agent.train()
        cnts, ates, grows, lens = zip(*deadInfo)
        cnts = sum(cnts) / len(cnts)
        ates = sum(ates) / len(ates)
        grows = sum(grows) / len(grows)
        lens = sum(lens) / len(lens)
        maxGrows = max(maxGrows, grows + 10)
        print(f'iter: {iter}, epsilon: {cfg.agent.epsilon:.2f}, survival: {cnts:.2f}, ate: {ates:.2f}, grow: {grows:.2f}, len: {lens:.2f}')

        with writer.as_default():
            tf.summary.scalar('life time', cnts, iter)
            tf.summary.scalar('ate count', ates, iter)
            tf.summary.scalar('grow count', grows, iter)
            tf.summary.scalar('body length', lens, iter)

        if iter % 5 == 0:
            cfg.agent.model.save('model/model.h5')