import tensorflow as tf

from agent import PPOAgent
from game import Game

# state, action, prob, value, reward, done

if __name__ == '__main__':
    games = [Game() for _ in range(128)]
    agent = PPOAgent()
    iter = 0

    while True:
        states = []

        for game in games:
            gameState = agent.getGameState(game.gridWidth, game.gridHeight, game.snakes, game.foodManager.foods)

            for snake in game.snakes:
                states.append(agent.getState(snake, game.gridWidth, game.gridHeight, gameState, game.foodManager.foods))

        probs, values = agent.model(tf.stack(states))
        actions = tf.squeeze(tf.random.categorical(tf.math.log(probs), 1))
        probs = tf.gather(probs, actions, batch_dims=1).numpy().tolist()
        actions = actions.numpy().tolist()
        values = tf.squeeze(values).numpy().tolist()

        for i, (game, snake) in enumerate((game, snake) for game in games for snake in game.snakes):
            action, prob, value = actions[i], probs[i], values[i]
            snake.rememberStageA(states[i], action, prob, value)
            snake.setDirection(action - 1)

        for game in games: game.tick(agent)

        agent.train()