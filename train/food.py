class FoodManager:

    def __init__(self, game):
        self.maxFoodCount = game.gridHeight * game.gridWidth // 50
        self.minFoodCount = 3
        self.deltaTime = 0
        self.game = game
        self.foods = []

    def generateFood(self):
        self.foods.append(self.game.getValidPos())

    def tick(self):
        foodRatio = (len(self.foods) - self.minFoodCount) / (self.maxFoodCount - self.minFoodCount)
        interval = 2 + 32 * foodRatio
        self.deltaTime += 1

        while len(self.foods) < self.maxFoodCount and self.deltaTime >= interval:
            self.deltaTime -= interval
            self.foods.append(self.game.getValidPos())

        while len(self.foods) < self.minFoodCount:
            self.foods.append(self.game.getValidPos())
            self.deltaTime = 0