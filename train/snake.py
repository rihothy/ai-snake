from collections import deque
import tensorflow as tf
import random

deadInfo = deque(maxlen=5000)

class Snake:

    zeroState = tf.zeros((23, 23, 5), 'float32')

    def __init__(self, game):
        x, y = game.getValidPos()

        self.body = [(x, y) for _ in range(random.randint(3, 32))]
        # self.body = [(x, y) for _ in range(3)]
        self.survivalCount = 0
        self.foodCount = 0
        self.direction = 0
        self.growCount = 0
        self.ateCount = 0
        self.alive = True
        self.ate = False

        self.trajectory = []

    def rememberStageA(self, state, action, value):
        self.state, self.action, self.value = state, action, value

    def rememberStageB(self, agent):
        if not hasattr(self, 'state'):
            return

        steps = agent.steps
        gamma = agent.gamma

        reward = -0.1
        if self.ate: reward = 1
        if not self.alive: reward = -2.5

        for i in range(-1, -steps, -1):
            if len(self.trajectory) >= abs(i):
                self.trajectory[i][3] += (gamma ** abs(i)) * reward

        if len(self.trajectory) >= steps:
            self.trajectory[-steps][4] = self.state
            self.trajectory[-steps][5] = self.value
            self.trajectory[-steps][6] = False

        #     0,      1,     2,      3,         4,         5,    6
        # state, action, value, reward, nextState, nextValue, done
        self.trajectory.append([self.state, self.action, self.value, reward, Snake.zeroState, 0, True])

        if not self.alive:
            for i in range(len(self.trajectory)):
                if i < len(self.trajectory) - 128:
                    if random.random() > ((i / (len(self.trajectory) - 128)) ** 2) * 0.95 + 0.05:
                        continue

                state, action, value, reward, nextState, nextValue, done = self.trajectory[i]
                agent.memory.tempPush(state, action, reward, nextState, done, (abs(reward + (gamma ** steps) * nextValue - value) + 1e-6) ** agent.memory.alpha)

    def setDirection(self, action):
        self.direction = (self.direction + action + 4) % 4
        self.survivalCount += 1
        self.ate = False

    def move(self):
        x, y = self.body[0]

        self.body.insert(0, (x + [1, 0, -1, 0][self.direction], y + [0, 1, 0, -1][self.direction]))
        self.body.pop()

    def checkCollision(self, gridWidth, gridHeight, snakes):
        hx, hy = self.body[0]

        if hx < 0 or hx >= gridWidth or hy < 0 or hy >= gridHeight:
            deadInfo.append((self.survivalCount, self.ateCount, self.growCount, len(self.body)))
            self.alive = False
            return

        for x, y in self.body[1:]:
            if x == hx and y == hy:
                deadInfo.append((self.survivalCount, self.ateCount, self.growCount, len(self.body)))
                self.alive = False
                return

        for otherSnake in snakes:
            if otherSnake is not self:
                for x, y in otherSnake.body:
                    if x == hx and y == hy:
                        deadInfo.append((self.survivalCount, self.ateCount, self.growCount, len(self.body)))
                        self.alive = False
                        return

    def checkFood(self, foods):
        if not self.alive:
            return foods

        hx, hy = self.body[0]

        for x, y in foods:
            if hx == x and hy == y:
                self.foodCount += 1
                self.ateCount += 1
                self.ate = True

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

        return list(filter(lambda f: f[0] != hx or f[1] != hy, foods))
