from collections import deque
import tensorflow as tf

class Snake:

    zeroState = tf.zeros((23, 23, 5), 'float32')
    deadInfo = deque(maxlen=500)

    def __init__(self, game):
        x, y = game.getValidPos()

        self.body = [(x, y) for _ in range(3, 20)]
        self.survivalCount = 0
        self.foodCount = 0
        self.direction = 0
        self.growCount = 0
        self.ateCount = 0
        self.alive = True
        self.ate = False

        self.states = []
        self.trajectory = []

    def rememberStageA(self, state, action, prob, value):
        self.state, self.action, self.prob, self.value = state, action, prob, value
        self.states.append(state)

    def rememberStageB(self, agent):
        if not hasattr(self, 'state'):
            return

        reward = (1 if self.ate else -0.1) if self.alive else -5
        self.trajectory.append((self.state, self.action, self.prob, self.value, reward, not self.alive))

        if not self.alive:
            deltas = []
            advs = [0]

            for i, (_, _, _, value, reward, done) in enumerate(self.trajectory):
                deltas.append(reward + (0 if done else agent.gamma * self.trajectory[i + 1][3]) - value)

            for delta, (*_, done) in zip(deltas[::-1], self.trajectory[::-1]):
                advs.append(delta + agent.gamma * agent.lambd * advs[-1] * ~done)

            advs = advs[::-1][:-1]

            for adv, (state, action, prob, value, *_) in zip(advs, self.trajectory):
                agent.buffer.append((state, adv, action, prob, adv + value))

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
            Snake.deadInfo.append((self.survivalCount, self.ateCount, self.growCount, len(self.body)))
            self.alive = False
            return

        for x, y in self.body[1:]:
            if x == hx and y == hy:
                Snake.deadInfo.append((self.survivalCount, self.ateCount, self.growCount, len(self.body)))
                self.alive = False
                return

        for otherSnake in snakes:
            if otherSnake is not self:
                for x, y in otherSnake.body:
                    if x == hx and y == hy:
                        Snake.deadInfo.append((self.survivalCount, self.ateCount, self.growCount, len(self.body)))
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