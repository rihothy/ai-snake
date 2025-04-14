from tensorflow import keras
import tensorflow as tf
import numpy as np
import random
import math

from snake import Snake

class PPOAgent:

    def __init__(self):
        self.iter = 0
        self.steps = 3
        self.gamma = 0.99
        self.lambd = 0.95
        self.viewSize = 23
        self.stateSize = 5
        self.actionSize = 3
        self.maxGradNorm = 0.5
        self.entropyCoef = 0.01

        self.buffer = []
        self.writer = tf.summary.create_file_writer('log/')

        self.model = self.createModel()

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

        prob = keras.layers.Dense(self.actionSize, 'softmax')(keras.layers.Dense(64, 'relu')(layer))
        value = keras.layers.Dense(1, 'linear')(keras.layers.Dense(64, 'relu')(layer))

        model = keras.Model(input, [prob, value])
        model.compile(keras.optimizers.Adam(2.5e-4))

        return model

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

    def train(self, epochs=4, sampleSize=4096, batchSize=32):
        if len(self.buffer) < sampleSize:
            return

        self.iter += 1
        random.shuffle(self.buffer)
        sampleSize = len(self.buffer) // batchSize * batchSize
        self.buffer = self.buffer[:sampleSize]

        state, adv, action, prob, value = zip(*self.buffer)
        self.buffer = []

        state = tf.stack(state)
        adv = tf.stack(adv)
        action = tf.stack(action)
        prob = tf.stack(prob)
        value = tf.stack(value)

        adv = (adv - tf.reduce_mean(adv)) / (tf.math.reduce_std(adv) + 1e-6)

        for _ in range(epochs):
            for i in range(0, sampleSize, batchSize):
                with tf.GradientTape() as tape:
                    newProb, newValue = self.model(state[i:i+batchSize])
                    entropyLoss = tf.reduce_mean(-tf.reduce_sum(newProb * tf.math.log(newProb + 1e-10), axis=1))

                    newProb = tf.gather(newProb, action[i:i+batchSize], batch_dims=1)
                    ratio = tf.exp(tf.math.log(newProb) - tf.math.log(prob[i:i+batchSize]))
                    ppoLoss = tf.reduce_mean(-tf.minimum(ratio * adv[i:i+batchSize], tf.clip_by_value(ratio, 0.8, 1.2) * adv[i:i+batchSize]))

                    valueLoss = keras.losses.mse(value[i:i+batchSize], newValue)

                    loss = ppoLoss - self.entropyCoef * entropyLoss + 0.5 * valueLoss

                    gradients = tape.gradient(loss, self.model.trainable_weights)
                    gradients, _ = tf.clip_by_global_norm(gradients, self.maxGradNorm)
                    self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))

        with self.writer.as_default():
            cnts, ates, *_ = zip(*Snake.deadInfo)
            cnts, ates = sum(cnts) / len(cnts), sum(ates) / len(ates)
            print(f'iter: {self.iter}, survival: {cnts:.2f}, ate: {ates:.2f}')

            tf.summary.scalar('life time', cnts, self.iter)
            tf.summary.scalar('ate count', ates, self.iter)

        if self.iter % 10 == 0:
            self.model.save('model/model.h5')