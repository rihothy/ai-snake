from collections import deque
import tensorflow as tf
import numpy as np
import random
import math

import config as cfg

deadInfo = deque(maxlen=500)
getNonlinearProb = lambda x: ((math.sin((x ** 4) * math.pi - math.pi / 2) + 1) / 2 + 0.05) / 1.05

class Snake:

    zeroState = None

    def __init__(self, length):
        while True:
            x = random.randint(0, cfg.gridWidth - 1)
            y = random.randint(0, cfg.gridHeight - 1)

            if not cfg.checkPositionOccupied(x, y):
                break

        self.body = [(x, y) for _ in range(3 + round(length * (1 - getNonlinearProb(random.random()))))]
        self.direction = self.nextDirection = 0
        self.survivalCount = 0
        self.foodCount = 0
        self.growCount = 0
        self.ateCount = 0
        self.alive = True
        self.ate = False

        self.trajectory = []

    def rememberStageA(self, state, action, value):
        self.state, self.action, self.value = state, action, value

    def rememberStageB(self):
        if not hasattr(self, 'state'):
            return

        if Snake.zeroState is None:
            Snake.zeroState = tf.zeros_like(self.state)

        steps = cfg.agent.steps
        gamma = cfg.agent.gamma

        reward, mode = -0.1, 0
        if self.ate: reward, mode = reward + 2, 1
        if not self.alive: reward, mode = reward - 2, 2

        for i in range(-1, -steps, -1):
            if len(self.trajectory) >= abs(i):
                self.trajectory[i][3] += (gamma ** abs(i)) * reward

        if len(self.trajectory) >= steps:
            self.trajectory[-steps][4] = self.state
            self.trajectory[-steps][5] = self.value
            self.trajectory[-steps][6] = False

        #     0,      1,     2,      3,         4,         5,    6     7
        # state, action, value, reward, nextState, nextValue, done, mode
        self.trajectory.append([self.state, self.action, self.value, reward, Snake.zeroState, 0, True, mode])

        if not self.alive:
            for i in range(len(self.trajectory)):
                state, action, value, reward, nextState, nextValue, done, mode = self.trajectory[i]

                if mode or i >= len(self.trajectory) - 50 or random.random() < getNonlinearProb((i + 1) / (len(self.trajectory) - 50)):
                    if cfg.gridWidth != cfg.gridHeight and random.random() < 0.5:
                        state, nextState, action = np.rot90(state), np.rot90(nextState), (action + 3) % 4

                    cfg.agent.memory.push(state, action, reward, nextState, done, (abs(reward + (gamma ** steps) * nextValue - value) + 1e-6) ** cfg.agent.memory.alpha)

    def setDirection(self, newDirection):
        self.survivalCount += 1
        self.ate = False

        if (self.direction - newDirection + 4) % 4 != 2:
            self.nextDirection = newDirection

    def move(self):
        x, y = self.body[0]

        self.direction = self.nextDirection
        self.body.insert(0, (x + [1, 0, -1, 0][self.direction], y + [0, 1, 0, -1][self.direction]))
        self.body.pop()

    def checkCollision(self):
        hx, hy = self.body[0]

        if hx < 0 or hx >= cfg.gridWidth or hy < 0 or hy >= cfg.gridHeight:
            deadInfo.append((self.survivalCount, self.ateCount, self.growCount, len(self.body)))
            self.alive = False
            return

        for x, y in self.body[1:]:
            if x == hx and y == hy:
                deadInfo.append((self.survivalCount, self.ateCount, self.growCount, len(self.body)))
                self.alive = False
                return

        for snake in cfg.snakes:
            if snake is not self:
                for x, y in snake.body:
                    if x == hx and y == hy:
                        deadInfo.append((self.survivalCount, self.ateCount, self.growCount, len(self.body)))
                        self.alive = False
                        return

    def checkFood(self):
        if not self.alive:
            return

        hx, hy = self.body[0]

        for x, y in cfg.foodManager.foods:
            if hx == x and hy == y:
                self.foodCount += 1
                self.ateCount += 1
                self.ate = True

        cfg.foodManager.foods = list(filter(lambda f: f[0] != hx or f[1] != hy, cfg.foodManager.foods))

        if self.ate:
            def getRequireFood():
                if len(self.body) < 10:
                    return 1
                elif len(self.body) < 20:
                    return 2
                elif len(self.body) < 30:
                    return 3
                elif len(self.body) < 40:
                    return 4
                else:
                    return 5

            while self.foodCount >= getRequireFood():
                self.foodCount -= getRequireFood()
                self.body.append(self.body[-1])
                self.growCount += 1