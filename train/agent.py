from tensorflow import keras
import tensorflow as tf
import numpy as np
import random
import math

class ExperienceReplay:

    def __init__(self, maxSize = 100000):
        self.maxSize = maxSize
        self.tempMemories = []
        self.memories = []
        self.priors = []
        self.newSize = 0
        self.alpha = 0.6
        self.beta = 0.4

    def tempPush(self, state, action, reward, nextState, done, prior):
        self.tempMemories.append((state, action, reward, nextState, done, prior))
        self.newSize += 1

    def push(self, size):
        random.shuffle(self.tempMemories)

        for _ in range(size):
            state, action, reward, nextState, done, prior = self.tempMemories.pop()
            self.newSize -= 1

            self.memories.append((state, action, reward, nextState, done))
            self.priors.append(prior)

            if len(self.memories) > self.maxSize:
                self.memories.pop(0)
                self.priors.pop(0)

    def sample(self, size):
        indices = random.choices(range(len(self.memories)), self.priors, k=size)
        totPrior = sum(self.priors)
        samples = []
        priors = []

        for index in indices:
            samples.append(self.memories[index])
            priors.append(self.priors[index] / totPrior)

        return samples, indices, (len(self.memories) * tf.convert_to_tensor(priors, 'float32')) ** (-self.beta)


class DQNAgent:

    def __init__(self):
        self.steps = 3
        self.gamma = 0.95
        self.viewSize = 23
        self.epsilon = 1
        self.stateSize = 5
        self.actionSize = 3
        self.epsilonMin = 0.01
        self.epsilonDecay = 0.9995
        self.memory = ExperienceReplay()

        self.model = self.createModel()
        self.targetModel = self.createModel()
        self.targetModel.set_weights(self.model.get_weights())

        self.optimizer = keras.optimizers.Adam()

    def createModel(self):
        input = layer = keras.Input((self.viewSize, self.viewSize, self.stateSize))
        layer = keras.layers.Conv2D(32, 3, padding='same', activation='relu')(layer)
        layer = keras.layers.Conv2D(64, 3, 2, padding='same', activation='relu')(layer)
        layer = keras.layers.Conv2D(64, 3, padding='same', activation='relu')(layer)
        layer = keras.layers.Conv2D(64, 3, 2, padding='same', activation='relu')(layer)
        layer = keras.layers.Conv2D(64, 3, padding='same', activation='relu')(layer)
        layer = keras.layers.Conv2D(128, 3, 2, padding='same', activation='relu')(layer)
        layer = keras.layers.Conv2D(128, 3, activation='relu')(layer)

        layer = keras.layers.Flatten()(layer)
        layer = keras.layers.Dense(128, 'relu')(layer)

        v = keras.layers.Dense(64, 'relu')(layer)
        v = keras.layers.Dense(1, 'linear')(v)

        a = keras.layers.Dense(64, 'relu')(layer)
        a = keras.layers.Dense(self.actionSize, 'linear')(a)

        model = keras.Model(input, [v, a])
        model.compile()

        return model

    def forward(self, model, state):
        v, a = model(state)
        return v + (a - tf.reduce_mean(a, -1, True))

    def getGameState(self, gridWidth, gridHeight, snakes, foods):
        state = np.zeros((gridHeight, gridWidth, 2), 'float32')

        for snake in snakes:
            for i, (x, y) in enumerate(snake.body[::-1]):
                state[y, x, 0] = 1 - 0.75 * i / len(snake.body)

        for x, y in foods:
            state[y, x, 1] = 1

        return state

    def getState(self, snake, gridWidth, gridHeight, gameState, foods):
        state = np.zeros((self.viewSize, self.viewSize, self.stateSize), 'float32')
        halfViewSize = self.viewSize // 2
        hx, hy = snake.body[0]

        offsetX = halfViewSize - hx
        offsetY = halfViewSize - hy

        lvy, rvy = max(offsetY, 0), min(gridHeight + offsetY, self.viewSize)
        lvx, rvx = max(offsetX, 0), min(gridWidth + offsetX, self.viewSize)

        lwy, rwy = max(0, hy - halfViewSize), min(gridHeight, hy + halfViewSize + 1)
        lwx, rwx = max(0, hx - halfViewSize), min(gridWidth, hx + halfViewSize + 1)

        for i, (x, y) in enumerate(snake.body[::-1]):
            x += offsetX
            y += offsetY

            if x >= 0 and x < self.viewSize and y >= 0 and y < self.viewSize:
                state[y, x, 0] = 1 - 0.75 * i / len(snake.body)

        state[lvy:rvy, lvx:rvx, 1] = gameState[lwy:rwy, lwx:rwx, 0]

        if not (rvy - lvy == self.viewSize and rvx - lvx == self.viewSize):
            state[:, :, 2] = 1
            state[lvy:rvy, lvx:rvx, 2] = 0

        farFoods = {}

        state[lvy:rvy, lvx:rvx, 3] = gameState[lwy:rwy, lwx:rwx, 1]

        for x, y in foods:
            x += offsetX
            y += offsetY

            if x < 0 or x >= self.viewSize or y < 0 or y >= self.viewSize:
                x -= halfViewSize
                y -= halfViewSize

                radius = math.sqrt(x * x + y * y)

                x = round(x / radius * halfViewSize) + halfViewSize
                y = round(y / radius * halfViewSize) + halfViewSize

                if x >= 0 and x < self.viewSize and y >= 0 and y < self.viewSize:
                    if (x, y) not in farFoods:
                        farFoods[(x, y)] = 0

                    farFoods[(x, y)] += (2 - min(2, (radius - halfViewSize) / halfViewSize)) / 10

        for (x, y), val in farFoods.items():
            state[y, x, 4] = min(1, val)

        return tf.convert_to_tensor(np.rot90(state, snake.direction))

    def train(self, sampleSize = 1024, batchSize = 32):
        if len(self.memory.memories) < sampleSize:
            return

        samples, indices, weight = self.memory.sample(sampleSize)
        state, action, reward, nextState, done = zip(*samples)

        state = tf.stack(state)
        action = tf.stack(action)
        reward = tf.stack(reward)
        nextState = tf.stack(nextState)
        done = tf.cast(tf.stack(done), 'float32')

        nextQ = self.forward(self.targetModel, nextState)
        nextAction = tf.argmax(self.forward(self.model, nextState), -1)

        targetQ = reward + (1 - done) * (self.gamma ** self.steps) * tf.gather_nd(nextQ, tf.transpose([tf.range(sampleSize, dtype='int64'), nextAction]))

        for i in range(0, sampleSize, batchSize):
            with tf.GradientTape() as tape:
                y_pred = tf.gather_nd(self.forward(self.model, state[i:i+batchSize]), tf.stack([tf.range(batchSize), action[i:i+batchSize]], 1))
                y_true = targetQ[i:i+batchSize]

                error = y_pred - y_true
                abs_error = tf.abs(error)
                loss = tf.reduce_mean(tf.where(abs_error <= 1, 0.5 * tf.square(error), abs_error - 0.5) * weight[i:i+batchSize])

            gradients = tape.gradient(loss, self.model.trainable_weights)
            gradients, _ = tf.clip_by_global_norm(gradients, 10)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))

            priors = tf.pow(tf.abs(y_true - y_pred) + 1e-6, self.memory.alpha).numpy()

            for j, index in enumerate(indices[i:i+batchSize]):
                self.memory.priors[index] = priors[j]

        self.epsilon = max(self.epsilon * self.epsilonDecay, self.epsilonMin)
        self.targetModel.set_weights([w1 * 0.25 + w2 * 0.75 for w1, w2 in zip(self.model.get_weights(), self.targetModel.get_weights())])