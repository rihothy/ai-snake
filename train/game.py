import tensorflow as tf
import random

from snake import Snake, deadInfo
from food import FoodManager
from agent import DQNAgent

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


if __name__ == '__main__':
    writer = tf.summary.create_file_writer('log/')
    games = [Game() for _ in range(128)]
    agent = DQNAgent()
    iter = 0

    while True:
        directions = []
        states = []

        for game in games:
            gameState = agent.getGameState(game.gridWidth, game.gridHeight, game.snakes, game.foodManager.foods)

            for snake in game.snakes:
                states.append(agent.getState(snake, game.gridWidth, game.gridHeight, gameState, game.foodManager.foods))
                directions.append(snake.direction)

        values = agent.forward(agent.model, tf.stack(states))
        actions = tf.argmax(values, -1).numpy()
        values = values.numpy()

        for i, (game, snake) in enumerate((game, snake) for game in games for snake in game.snakes):
            if random.random() < agent.epsilon:
                invalidActions, validActions = [], []

                for action in range(3):
                    direction = (snake.direction + action + 3) % 4
                    x, y = snake.body[0][0] + [1, 0, -1, 0][direction], snake.body[0][1] + [0, 1, 0, -1][direction]

                    if x < 0 or x >= game.gridWidth or y < 0 or y >= game.gridHeight or any(any((tx == x and ty == y) for tx, ty in otherSnake.body[:-1]) for otherSnake in game.snakes):
                        invalidActions.append(action)
                    else:
                        validActions.append(action)

                action = random.randint(0, 2) if len(validActions) == 0 else random.choice(validActions)

                for invalidAction in invalidActions:
                    if invalidAction != action:
                        prior = (abs(-5 - float(values[i, invalidAction])) + 1e-6) ** agent.memory.alpha
                        agent.memory.tempPush(states[i], invalidAction, -5, Snake.zeroState, True, prior)
            else:
                action = int(actions[i])

            snake.rememberStageA(states[i], action, float(values[i, action]))
            snake.setDirection(action - 1)

        for game in games: game.tick(agent)

        if len(agent.memory.memories) + agent.memory.newSize >= 1024 and agent.memory.newSize >= 512:
            agent.memory.push(512)
            agent.train()

            iter += 1
            cnts, ates, grows, lens = zip(*deadInfo)
            cnts, ates = sum(cnts) / len(cnts), sum(ates) / len(ates)
            print(f'iter: {iter}, epsilon: {agent.epsilon:.2f}, survival: {cnts:.2f}, ate: {ates:.2f}')

            with writer.as_default():
                tf.summary.scalar('life time', cnts, iter)
                tf.summary.scalar('ate count', ates, iter)

            if iter % 5 == 0:
                agent.model.save('model/model.h5')