import tensorflow as tf
import numpy as np
import random

from snake import Snake, deadInfo
from agent import DQNAgent
from game import Game

if __name__ == '__main__':
    writer = tf.summary.create_file_writer('log/')
    games = [Game(32, 64) for _ in range(128)]
    agent = DQNAgent()
    iter = 0

    agent.epsilon = 0.05
    agent.model.load_weights('model/model3.h5')
    agent.targetModel.load_weights('model/model3.h5')

    while True:
        states = []

        for game in games:
            gameState = agent.getGameState(game.gridWidth, game.gridHeight, game.snakes, game.foodManager.foods)

            for snake in game.snakes:
                states.append(agent.getState(snake, game.gridWidth, game.gridHeight, gameState, game.foodManager.foods))

        values = agent.forward(agent.model, np.stack(states))
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
                        prior = (abs(-2.5 - float(values[i, invalidAction])) + 1e-6) ** agent.memory.alpha
                        agent.memory.tempPush(states[i], invalidAction, -2.5, Snake.zeroState, True, prior)
            else:
                action = int(actions[i])

            snake.rememberStageA(states[i], action, float(values[i, action]))
            snake.setDirection(action - 1)

        for game in games: game.tick()

        for game in games: game.postTick(agent)

        if len(agent.memory.memories) + agent.memory.newSize >= 75000 and agent.memory.newSize >= 512:
            agent.memory.push(512)
            agent.train()
            iter += 1

            with writer.as_default():
                cnts, ates, _, _ = zip(*deadInfo)
                max_cnt, max_ate = max(cnts), max(ates)
                cnts, ates = sum(cnts) / len(cnts), sum(ates) / len(ates)
                print(f'iter: {iter}, epsilon: {agent.epsilon:.2f}, survival: {cnts:.2f}, ate: {ates:.2f}, max survival: {max_cnt}, max ate: {max_ate}')

                tf.summary.scalar('life time', cnts, iter)
                tf.summary.scalar('ate count', ates, iter)

            if iter % 100 == 0:
                agent.model.save('model/model4.h5')