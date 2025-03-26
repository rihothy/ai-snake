class ExperienceReplay {
    constructor(maxSize = 10000) {
        this.maxSize = maxSize;
        this.memories = [];
    }

    push(state, action, reward, nextState, done) {
        if (this.memories.length >= this.maxSize) {
            tf.dispose(this.memories.shift());
        }

        this.memories.push({state, action, reward, nextState, done});
    }

    sample(size) {
        const samples = [];

        for (let i = 0; i < size; i++) {
            samples.push(this.memories[Math.floor(Math.random() * this.memories.length)]);
        }

        return samples;
    }
}

class DQNAgent {
    constructor(savePath) {
        this.gamma = 0.95;
        this.viewSize = 21;
        this.epsilon = 1.0;
        this.actionSize = 4;
        this.epsilonMin = 0.01;
        this.epsilonDecay = 0.995;
        this.memory = new ExperienceReplay();

        this.savePath = savePath;
        this.model = this.createModel();
        this.targetModel = this.createModel();
        this.targetModel.setWeights(this.model.getWeights());
    }

    createModel() {
        let model = tf.sequential();

        model.add(tf.layers.conv2d({filters: 32, kernelSize: 3, activation: 'relu', padding: 'same', inputShape: [this.viewSize, this.viewSize, 5]}));
        model.add(tf.layers.conv2d({filters: 64, strides: 2, kernelSize: 3, activation: 'relu'}));
        model.add(tf.layers.conv2d({filters: 64, kernelSize: 3, padding: 'same', activation: 'relu'}));
        model.add(tf.layers.conv2d({filters: 64, strides: 2, kernelSize: 3, activation: 'relu'}));
        model.add(tf.layers.conv2d({filters: 128, kernelSize: 4, activation: 'relu'}));
        model.add(tf.layers.flatten());
        model.add(tf.layers.dense({units: 256, activation: 'relu'}));
        model.add(tf.layers.dense({units: this.actionSize, activation: 'linear'}));

        model.compile({optimizer: tf.train.adam(0.001), loss: 'meanSquaredError'});

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

        for (const other of snakes) {
            if (other !== snake && other.alive) {
                for (let i = other.body.length - 1; i >= 0; i--) {
                    const x = other.body[i].x + offsetX;
                    const y = other.body[i].y + offsetY;

                    if (x >= 0 && x < this.viewSize && y >= 0 && y < this.viewSize) {
                        state.set(1 - 0.75 * i / snake.body.length, x, y, 2);
                    }
                }
            }
        }

        for (let y = 0; y < this.viewSize; y++) {
            for (let x = 0; x < this.viewSize; x++) {
                const worldX = x - offsetX;
                const worldY = y - offsetY;

                if (worldX < 0 || worldX >= gridWidth || worldY < 0 || worldY >= gridHeight) {
                    state.set(1, x, y, 3);
                }
            }
        }

        for (const food of foodManager.foods) {
            const x = food.x + offsetX;
            const y = food.y + offsetY;

            if (x >= 0 && x < this.viewSize && y >= 0 && y < this.viewSize) {
                state.set(1, x, y, 4);
            }
        }

        return state.toTensor();
    }

    getAction(state, epsilon = null) {
        return tf.tidy(() => {
            return Math.random() < (epsilon === null ? this.epsilon : epsilon) ? Math.floor(Math.random() * this.actionSize) : this.model.predict(state.expandDims(0)).argMax(1).dataSync()[0];
        });
    }

    async train(sampleSize = 2048, batchSize = 32) {
        if (this.memory.memories.length < sampleSize) {
            return;
        }

        const samples = this.memory.sample(sampleSize);
        const states = [], nextStates = [];

        for (const sample of samples) {
            states.push(sample.state.arraySync());
            nextStates.push(sample.nextState.arraySync());
        }

        const stateTensor = tf.tensor4d(states);
        const nextStateTensor = tf.tensor4d(nextStates);

        const targetQ = this.targetModel.predict(nextStateTensor);
        const currentQ = this.model.predict(stateTensor);

        const targets = currentQ.arraySync();

        for (let i = 0; i < sampleSize; i++) {
            targets[i][samples[i].action] = samples[i].reward + (samples[i].done ? 0 : this.gamma * Math.max(...targetQ.arraySync()[i]))
        }

        const targetTensor = tf.tensor2d(targets);

        await this.model.fit(stateTensor, targetTensor, {epochs: 1, batchSize});

        if (this.savePath) {
            await this.model.save(this.savePath);
        }

        tf.dispose([stateTensor, nextStateTensor, targetQ, currentQ, targetTensor]);
        this.epsilon = Math.max(this.epsilonMin, this.epsilon * this.epsilonDecay);
        this.targetModel.setWeights(this.model.getWeights());
    }
}