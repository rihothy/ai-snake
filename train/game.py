import random

from food import FoodManager
from snake import Snake

class Game:

    def __init__(self, gridWidth=32, gridHeight=32):
        self.gridWidth = gridWidth
        self.gridHeight = gridHeight

        self.snakes: list[Snake] = [] 
        self.foodManager = FoodManager(self)

        self.delayFoods = []
        self.delaySnakes = []

        for _ in range(5):
            self.foodManager.generateFood()

        for _ in range(5):
            self.snakes.append(Snake(self))

    def isValidPos(self, x, y):
        return x >= 0 and x < self.gridWidth and y >= 0 and y < self.gridHeight and not any(any((tx == x and ty == y) for tx, ty in snake.body) for snake in self.snakes) and not any((tx == x and ty == y) for tx, ty in self.foodManager.foods)

    def getValidPos(self):
        while True:
            x, y = random.randint(0, self.gridWidth - 1), random.randint(0, self.gridHeight - 1)

            if self.isValidPos(x, y):
                return x, y

    def tick(self, agent):
        for snake in self.snakes: snake.move()
        for snake in self.snakes: snake.checkCollision(self.gridWidth, self.gridHeight, self.snakes)
        for snake in self.snakes: self.foodManager.foods = snake.checkFood(self.foodManager.foods)

        for i in range(len(self.delayFoods)):
            self.delayFoods[i][-1] -= 1

            if self.delayFoods[i][-1] == 0 and self.isValidPos(*self.delayFoods[i][0]):
                self.foodManager.foods.append(self.delayFoods[i][0])

        for i in range(len(self.delaySnakes)):
            self.delaySnakes[i] -= 1

            if self.delaySnakes[i] == 0:
                self.snakes.append(Snake(self))

        self.foodManager.tick()
        self.delayFoods = list(filter(lambda x: x[-1], self.delayFoods))
        self.delaySnakes = list(filter(lambda x: x, self.delaySnakes))

        for snake in self.snakes:
            snake.rememberStageB(agent)

            if not snake.alive:
                for i in range(0, len(snake.body), 2):
                    self.delayFoods.append([snake.body[i], 6])

                self.delaySnakes.append(20)

        self.snakes = list(filter(lambda snake: snake.alive, self.snakes))