class FoodManager:

    def __init__(self, gridWidth, gridHeight):
        self.maxFoodCount = gridHeight * gridWidth // 50
        self.minFoodCount = 3
        self.deltaTime = 0

        self.delayFoods = []
        self.foods = []

    def tick(self, isValidPos, getValidPos):
        foodRatio = (len(self.foods) - self.minFoodCount) / (self.maxFoodCount - self.minFoodCount)
        interval = 2 + 32 * foodRatio
        self.deltaTime += 1

        for i in range(len(self.delayFoods)):
            self.delayFoods[i][0] -= 1

            if self.delayFoods[i][0] == 0 and isValidPos(*self.delayFoods[i][1]):
                self.foods.append(self.delayFoods[i][1])

        self.delayFoods = list(filter(lambda delayFood: delayFood[0], self.delayFoods))

        if len(self.foods) < self.maxFoodCount and self.deltaTime >= interval or len(self.foods) < self.minFoodCount:
            self.deltaTime = max(0, self.deltaTime - interval)
            self.foods.append(getValidPos())