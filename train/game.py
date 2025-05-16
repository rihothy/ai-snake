import random
import time
import os

from food import FoodManager
from snake import Snake

class Game:

    def __init__(self, gridWidth=32, gridHeight=32):
        self.gridWidth = gridWidth
        self.gridHeight = gridHeight

        self.snakes: list[Snake] = []
        self.foodManager = FoodManager(gridWidth, gridHeight)

        self.delaySnakes = []

        for _ in range(5):
            self.foodManager.foods.append(self.getValidPos())

        for _ in range(5):
            self.snakes.append(Snake(self))

    def isValidPos(self, x, y):
        return x >= 0 and x < self.gridWidth and y >= 0 and y < self.gridHeight and not any(any((tx == x and ty == y) for tx, ty in snake.body) for snake in self.snakes) and not any((tx == x and ty == y) for tx, ty in self.foodManager.foods)

    def getValidPos(self):
        while True:
            x, y = random.randint(0, self.gridWidth - 1), random.randint(0, self.gridHeight - 1)

            if self.isValidPos(x, y):
                return x, y

    def tick(self):
        for snake in self.snakes: snake.move()
        for snake in self.snakes: snake.checkCollision(self.gridWidth, self.gridHeight, self.snakes)
        for snake in self.snakes: self.foodManager.foods = snake.checkFood(self.foodManager.foods)

        for i in range(len(self.delaySnakes)):
            self.delaySnakes[i] -= 1

            if self.delaySnakes[i] == 0:
                self.snakes.append(Snake(self))

        self.foodManager.tick(lambda x, y: self.isValidPos(x, y), lambda: self.getValidPos())
        self.delaySnakes = list(filter(lambda x: x, self.delaySnakes))

    def postTick(self, agent):
        for snake in self.snakes:
            snake.rememberStageB(agent)

            if not snake.alive:
                for i in range(0, len(snake.body), 2):
                    self.foodManager.delayFoods.append([6, snake.body[i]])

                self.delaySnakes.append(20)

        self.snakes = list(filter(lambda snake: snake.alive, self.snakes))

    def display(self):
        os.system('clear')

        print('*' * (self.gridWidth * 2 + 2))
        for _ in range(self.gridHeight):
            print('*' + ' ' * self.gridWidth * 2 + '*')
        print('*' * (self.gridWidth * 2 + 2))

        for x, y in self.foodManager.foods:
            print(f'\033[{y + 2};{x * 2 + 2}H◼')

        for snake in self.snakes:
            for x, y in snake.body:
                print(f'\033[{y + 2};{x * 2 + 2}H◻')

        print(f'\033[{self.gridHeight + 2};0H')

        time.sleep(0.001)