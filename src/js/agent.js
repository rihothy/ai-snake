import { cfgs, vars, metricsInfo } from './global.js';
import * as tf from '@tensorflow/tfjs';

class ExperienceReplay {
    constructor(maxSize = 2000) {
        this.memories = [[], [], [], [], []];
        this.newPushSize = [0, 0, 0, 0, 0];
        this.maxSize = maxSize;
    }

    push(state, action, reward, nextState, done, type) {
        this.memories[type].push({state, action, reward, nextState, done});
        this.newPushSize[type]++;

        if (this.memories[type].length >= this.maxSize) {
            tf.dispose(this.memories[type].shift());
        }
    }

    isValid(minSize = 500, minTotNewPushSize = 1000) {
        for (let i = 0; i < this.memories.length; i++) {
            if (this.memories[i].length < minSize) {
                return false;
            }
        }

        if (this.newPushSize.reduce((a, b) => a + b, 0) < minTotNewPushSize) {
            return false;
        }

        return true;
    }

    sample(size) {
        const samples = [];

        for (let i = 0; i < size; i++) {
            let type = Math.floor(Math.random() * 3);

            if (type > 1) {
                type = 2 + Math.floor(Math.random() * 3);
            }

            samples.push(this.memories[type][Math.floor(Math.random() * this.memories[type].length)]);
        }

        this.newPushSize = [0, 0, 0, 0, 0];

        return samples;
    }
}

export class DQNAgent {
    constructor() {
        this.iter = 1;
        this.gamma = 0.95;
        this.viewSize = 23;
        this.epsilon = 1.0;
        this.actionSize = 4;
        this.epsilonMin = 0.01;
        this.epsilonDecay = 0.9975;
        this.memory = new ExperienceReplay();

        this.model = this.buildModel();
        this.targetModel = this.buildModel();
        this.targetModel.setWeights(this.model.getWeights());
    }

    static async create(modelPath) {
        const agent = new DQNAgent();

        if (modelPath) {
            agent.model = await tf.loadLayersModel(modelPath);
            agent.targetModel.setWeights(agent.model.getWeights());
        }

        agent.model.compile({optimizer: tf.train.adam(0.001), loss: 'meanSquaredError'});
        agent.model.summary();

        return agent;
    }

    buildModel() {
        const input = tf.layers.input({shape: [this.viewSize, this.viewSize, 5]});
        let layer = input;

        layer = tf.layers.conv2d({filters: 32, kernelSize: 3, padding: 'same', activation: 'relu'}).apply(layer);
        layer = tf.layers.conv2d({filters: 64, strides: 2, kernelSize: 3, padding: 'same', activation: 'relu'}).apply(layer);
        layer = tf.layers.conv2d({filters: 64, kernelSize: 3, padding: 'same', activation: 'relu'}).apply(layer);
        layer = tf.layers.conv2d({filters: 64, strides: 2, kernelSize: 3, padding: 'same', activation: 'relu'}).apply(layer);
        layer = tf.layers.conv2d({filters: 64, kernelSize: 3, padding: 'same', activation: 'relu'}).apply(layer);
        layer = tf.layers.conv2d({filters: 128, strides: 2, kernelSize: 3, padding: 'same', activation: 'relu'}).apply(layer);
        layer = tf.layers.conv2d({filters: 128, kernelSize: 3, activation: 'relu'}).apply(layer);
        layer = tf.layers.flatten().apply(layer);

        const sharedLayer = tf.layers.dense({units: 128, activation: 'relu'}).apply(layer);

        /* Dueling DQN
        // Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        const value = tf.layers.dense({units: 1}).apply(sharedFeatures);
        const advantage = tf.layers.dense({units: this.actionSize}).apply(sharedFeatures);
        const output = tf.layers.add([
            value,
            tf.layers.subtract([
                advantage,
                tf.mean(advantage, 1, true)
            ])
        ]);
        */

        const output = tf.layers.dense({units: this.actionSize}).apply(sharedLayer);
        const model = tf.model({inputs: input, outputs: output});

        return model;
    }

    getState(snake) {
        const state = tf.buffer([this.viewSize, this.viewSize, 5]);
        const halfViewSize = Math.floor(this.viewSize / 2);
        const head = snake.body[0];

        const offsetX = halfViewSize - head.x;
        const offsetY = halfViewSize - head.y;

        state.set(1, halfViewSize, halfViewSize, 0);

        for (let i = snake.body.length - 1; i >= 1; i--) {
            const x = snake.body[i].x + offsetX;
            const y = snake.body[i].y + offsetY;

            if (x >= 0 && x < this.viewSize && y >= 0 && y < this.viewSize) {
                state.set(1 - 0.75 * i / snake.body.length, x, y, 1);
            }
        }

        for (const otherSnake of vars.snakes) {
            if (otherSnake !== snake && otherSnake.states.alive) {
                for (let i = otherSnake.body.length - 1; i >= 0; i--) {
                    const x = otherSnake.body[i].x + offsetX;
                    const y = otherSnake.body[i].y + offsetY;

                    if (x >= 0 && x < this.viewSize && y >= 0 && y < this.viewSize) {
                        state.set(1 - 0.75 * i / otherSnake.body.length, x, y, 2);
                    }
                }
            }
        }

        for (let y = 0; y < this.viewSize; y++) {
            for (let x = 0; x < this.viewSize; x++) {
                const gridX = x - offsetX;
                const gridY = y - offsetY;

                if (gridX < 0 || gridX >= cfgs.gridWidth || gridY < 0 || gridY >= cfgs.gridHeight) {
                    state.set(1, x, y, 3);
                }
            }
        }

        for (const food of vars.foodManager.foods) {
            const x = food.x + offsetX;
            const y = food.y + offsetY;

            if (x >= 0 && x < this.viewSize && y >= 0 && y < this.viewSize) {
                state.set(1, x, y, 4);
            }
        }

        return state.toTensor();
    }

    getAction(state, epsilon) {
        return tf.tidy(() => {
            return Math.random() < (epsilon === undefined ? this.epsilon : epsilon) ? Math.floor(Math.random() * this.actionSize) : this.model.predict(state.expandDims(0)).argMax(1).dataSync()[0];
        });
    }

    async train(modelPath, sampleSize = 1024, batchSize = 32, epochs = 1) {
        const samples = this.memory.sample(sampleSize);
        const states = [], nextStates = [];

        for (const sample of samples) {
            states.push(sample.state.arraySync());
            nextStates.push(sample.nextState.arraySync());
        }

        const stateTensor = tf.tensor4d(states);
        const nextStateTensor = tf.tensor4d(nextStates);

        const targetTensor = tf.tidy(() => {
            const nextAction = this.model.predict(nextStateTensor).argMax(1).dataSync();
            const nextValue = this.targetModel.predict(nextStateTensor).arraySync();
            const targets = this.model.predict(stateTensor).arraySync();

            for (let i = 0; i < sampleSize; i++) {
                targets[i][samples[i].action] = samples[i].reward + (samples[i].done ? 0 : this.gamma * nextValue[i][nextAction[i]]);
            }

            return tf.tensor2d(targets);
        });

        await this.model.fit(stateTensor, targetTensor, {batchSize, epochs});

        if (modelPath) {
            await this.model.save(modelPath);
        }

        if (this.iter % 2 == 0) {
            this.targetModel.setWeights(this.model.getWeights());
        }

        this.epsilon = Math.max(this.epsilonMin, this.epsilon * this.epsilonDecay);
        tf.dispose([stateTensor, nextStateTensor, targetTensor]);

        let collideWallCount = 0, collideSelfCount = 0, collideOtherSnakeCount = 0;
        let avgLength = 0, avgLifeCount = 0;

        for (const info of metricsInfo) {
            if (info.type == 'collideWall') collideWallCount++;
            if (info.type == 'collideSelf') collideSelfCount++;
            if (info.type == 'collideOtherSnake') collideOtherSnakeCount++;

            avgLength += info.length;
            avgLifeCount += info.lifeCount;
        }

        console.log([
            `[iter: ${this.iter++}]`,
            `[gamma: ${this.gamma.toFixed(2)}]`,
            `[epsilon: ${this.epsilon.toFixed(2)}]`,
            `[avg length: ${(avgLength / metricsInfo.length).toFixed(2)}]`,
            `[avg life count: ${(avgLifeCount / metricsInfo.length).toFixed(2)}]`,
            `[collide wall: ${(collideWallCount / metricsInfo.length * 100).toFixed(2)}%]`,
            `[collide self: ${(collideSelfCount / metricsInfo.length * 100).toFixed(2)}%]`,
            `[collide other snake: ${(collideOtherSnakeCount / metricsInfo.length * 100).toFixed(2)}%]`
        ].join('\n'));
    }
};