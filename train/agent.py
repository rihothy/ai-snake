from tensorflow import keras
import tensorflow as tf
import numpy as np
import random
import math

import config as cfg

class ExperienceReplay:

    def __init__(self, maxSize = 100000):
        self.maxSize = maxSize
        self.memories = []
        self.priors = []
        self.newSize = 0
        self.alpha = 0.6
        self.beta = 0.4

    def push(self, state, action, reward, nextState, done, prior):
        self.memories.append((state, action, reward, nextState, done))
        self.priors.append(prior)
        self.newSize += 1

        if len(self.memories) > self.maxSize:
            self.memories.pop(0)
            self.priors.pop(0)

    def sample(self, size):
        indices = random.choices(range(len(self.memories)), self.priors, k=size)
        totPrior = sum(self.priors)
        self.newSize = 0
        samples = []
        priors = []

        for index in indices:
            samples.append(self.memories[index])
            priors.append(self.priors[index] / totPrior)

        return samples, indices, (len(self.memories) * tf.convert_to_tensor(priors, 'float32')) ** (-self.beta)


class NoisyDense(keras.layers.Layer):

    def __init__(self, units, activation):
        super(NoisyDense, self).__init__()
        self.units = units
        self.activation = keras.activations.get(activation)

    def get_config(self):
        config = super(NoisyDense, self).get_config()
        config.update({
            'units': self.units,
            'activation': tf.keras.activations.serialize(self.activation)
        })

        return config

    def build(self, input_shape):
        self.input_dim = input_shape[-1]
        self.mu_kernel = self.add_weight('mu_kernel', (self.input_dim, self.units), initializer='glorot_uniform', trainable=True)
        self.sigma_kernel = self.add_weight('sigma_kernel', (self.input_dim, self.units), initializer=keras.initializers.Constant(0.5), trainable=True)
        self.mu_bias = self.add_weight('mu_bias', (self.units,), initializer='zero', trainable=True)
        self.sigma_bias = self.add_weight('sigma_bias', (self.units,), initializer=keras.initializers.Constant(0.5), trainable=True)
        self.built = True

    def scale_noise(self, size):
        x = tf.random.normal((size,))
        return tf.sign(x) * tf.sqrt(tf.abs(x))

    def call(self, inputs):
        if True:
            epsilon_i = self.scale_noise(self.input_dim)
            epsilon_j = self.scale_noise(self.units)

            epsilon_kernel = tf.matmul(tf.expand_dims(epsilon_i, -1), tf.expand_dims(epsilon_j, 0))
            epsilon_bias = self.scale_noise(self.units)

            kernel = tf.add(self.mu_kernel, tf.multiply(self.sigma_kernel, epsilon_kernel))
            bias = tf.add(self.mu_bias, tf.multiply(self.sigma_bias, epsilon_bias))
        else:
            kernel = self.mu_kernel
            bias = self.mu_bias

        return self.activation(tf.add(tf.matmul(inputs, kernel), bias))


class DQNAgent:

    def __init__(self):
        self.steps = 3
        self.gamma = 0.95
        self.viewSize = 23
        self.epsilon = 1
        self.stateSize = 5
        self.actionSize = 4
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
        # layer = NoisyDense(128, 'relu')(layer)

        v = keras.layers.Dense(64, 'relu')(layer)
        v = keras.layers.Dense(1, 'linear')(v)
        # v = NoisyDense(64, 'relu')(layer)
        # v = NoisyDense(1, 'linear')(v)

        a = keras.layers.Dense(64, 'relu')(layer)
        a = keras.layers.Dense(self.actionSize, 'linear')(a)
        # a = NoisyDense(64, 'relu')(layer)
        # a = NoisyDense(self.actionSize, 'linear')(a)

        model = keras.Model(input, [v, a])
        model.compile()

        return model

    def forward(self, model, state):
        v, a = model(state)
        return v + (a - tf.reduce_mean(a, -1, True))

    def getState(self, snake):
        state = np.zeros((self.viewSize, self.viewSize, self.stateSize), 'float32')
        halfViewSize = self.viewSize // 2
        hx, hy = snake.body[0]

        offsetX = halfViewSize - hx
        offsetY = halfViewSize - hy

        def buildSnakeState(body, channel):
            l = 0

            for i in range(1, len(body)):
                if body[i] != body[i - 1]:
                    l += 1

            for i in range(len(body) - 1, -1, -1):
                x = body[i][0] + offsetX
                y = body[i][1] + offsetY

                if x >= 0 and x < self.viewSize and y >= 0 and y < self.viewSize:
                    state[y, x, channel] = 1 - 0.75 * min(i, l) / max(1, l)

        buildSnakeState(snake.body, 0)

        for otherSnake in cfg.snakes:
            if otherSnake is not snake and otherSnake.alive:
                buildSnakeState(otherSnake.body, 1)

        ly = max(offsetY, 0)
        ry = min(cfg.gridHeight + offsetY, self.viewSize)
        lx = max(offsetX, 0)
        rx = min(cfg.gridWidth + offsetX, self.viewSize)

        if not (ry - ly == self.viewSize and rx - lx == self.viewSize):
            state[:, :, 2] = 1
            state[ly:ry, lx:rx, 2] = 0

        farFoods = {}

        for x, y in cfg.foodManager.foods:
            x += offsetX
            y += offsetY

            if x >= 0 and x < self.viewSize and y >= 0 and y < self.viewSize:
                state[y, x, 3] = 1
            else:
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

        return tf.convert_to_tensor(state)

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