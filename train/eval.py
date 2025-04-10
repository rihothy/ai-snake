import tensorflow as tf
import random

from snake import Snake, deadInfo
from food import FoodManager
from agent import DQNAgent
import config as cfg

class SnakeAI:

    def __init__(self, snake):
        self.snake = snake


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


def reborn():
    newSnake = Snake()
    cfg.snakes.append(newSnake)
    aiControllers.append(SnakeAI(newSnake))


aiControllers = []
timerManager = TimerManager()
cfg.foodManager = FoodManager()
cfg.agent = DQNAgent()

for _ in range(5):
    cfg.foodManager.generateFood()

for _ in range(5):
    timerManager.setTimer(reborn, 1)


cfg.agent.model.load_weights('model.h5')
cfg.agent.model.summary()

while True:
    if len(aiControllers):
        actions = tf.argmax(cfg.agent.forward(cfg.agent.model, tf.stack([cfg.agent.getState(ai.snake) for ai in aiControllers])), -1).numpy()

        for i, ai in enumerate(aiControllers):
            ai.snake.setDirection(actions[i])

    for snake in cfg.snakes:
        snake.move()

    for snake in cfg.snakes:
        snake.checkCollision()

    for snake in cfg.snakes:
        snake.checkFood()

    timerManager.gameplayTick()
    cfg.foodManager.gameplayTick()

    for ai in aiControllers:
        snake = ai.snake

        if not snake.alive:
            for i in range(len(snake.body)):
                if i % 2 == 0:
                    def func(x, y):
                        return lambda: cfg.foodManager.generateFood(x, y)

                    timerManager.setTimer(func(snake.body[i][0], snake.body[i][1]), 6)

            timerManager.setTimer(reborn, 20)

            cfg.snakes.remove(snake)

    aiControllers = list(filter(lambda ai: ai.snake.alive, aiControllers))

    print(len(deadInfo))

    if len(deadInfo) >= 100:
        cnts, ates, lens = zip(*deadInfo)
        print(f'[avg] survival: {sum(cnts) / len(cnts):.2f}, ate: {sum(ates) / len(ates):.2f}, len: {sum(lens) / len(lens):.2f}, wrong: {sum([0 if c >= l else 1 for c, l in zip(cnts, lens)]) / len(lens) * 100:.2f}%')
        print(f'[min] survival: {min(cnts)}, ate: {min(ates)}, len: {min(lens)}')
        print(f'[max] survival: {max(cnts)}, ate: {max(ates)}, len: {max(lens)}')
        break