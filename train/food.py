import random

import config as cfg

class FoodManager:

    def __init__(self):
        self.maxFoodCount = cfg.gridHeight * cfg.gridWidth // 50
        self.minFoodCount = 3
        self.deltaTime = 0
        self.foods = []

    def generateFood(self, x = None, y = None):
        if x is not None and y is not None:
            if not cfg.checkPositionOccupied(x, y) and x >= 0 and x < cfg.gridWidth and y >= 0 and y < cfg.gridHeight:
                self.foods.append((x, y))
        else:
            while True:
                x = random.randint(0, cfg.gridWidth - 1)
                y = random.randint(0, cfg.gridHeight - 1)

                if not cfg.checkPositionOccupied(x, y):
                    self.foods.append((x, y))
                    break

    def gameplayTick(self):
        foodRatio = (len(self.foods) - self.minFoodCount) / (self.maxFoodCount - self.minFoodCount)
        interval = 2 + 32 * foodRatio
        self.deltaTime += 1

        while len(self.foods) < self.maxFoodCount and self.deltaTime >= interval:
            self.deltaTime -= interval
            self.generateFood()

        while len(self.foods) < self.minFoodCount:
            self.generateFood()
            self.deltaTime = 0