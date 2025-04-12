import { cfgs, vars } from './global.js';
import * as tf from '@tensorflow/tfjs';

export class DQNAgent {
    constructor() {
        this.viewSize = 23;
        this.stateSize = 5;
        this.actionSize = 4;
        this.model = undefined;
    }

    static async create(modelPath) {
        const agent = new DQNAgent();

        try {
            agent.model = await tf.loadLayersModel(modelPath);
        } catch(e) {
            return undefined;
        }

        agent.model.summary();

        return agent;
    }

    getState(snake) {
        const state = tf.buffer([this.viewSize, this.viewSize, this.stateSize]);
        const halfViewSize = Math.floor(this.viewSize / 2);
        const head = snake.body[0];

        const offsetX = halfViewSize - head.x;
        const offsetY = halfViewSize - head.y;

        const buildSnakeState = (body, channel) => {
            let len = 0;

            for (let i = 1; i < body.length; ++i) {
                if (body[i] != body[i - 1]) {
                    len += 1;
                }
            }

            for (let i = body.length - 1; i >= 0; --i) {
                const x = body[i].x + offsetX;
                const y = body[i].y + offsetY;
    
                if (x >= 0 && x < this.viewSize && y >= 0 && y < this.viewSize) {
                    state.set(1 - 0.75 * Math.min(i, len) / Math.max(1, len), y, x, channel);
                }
            }
        };

        buildSnakeState(snake.body, 0);

        for (const otherSnake of vars.snakes) {
            if (otherSnake !== snake && otherSnake.alive) {
                buildSnakeState(otherSnake.body, 1);
            }
        }

        for (let y = 0; y < this.viewSize; ++y) {
            for (let x = 0; x < this.viewSize; ++x) {
                const gridX = x - offsetX;
                const gridY = y - offsetY;

                if (gridX < 0 || gridX >= cfgs.gridWidth || gridY < 0 || gridY >= cfgs.gridHeight) {
                    state.set(1, y, x, 2);
                }
            }
        }

        let farFoods = {};

        for (const food of vars.foodManager.foods) {
            let x = food.x + offsetX;
            let y = food.y + offsetY;

            if (x >= 0 && x < this.viewSize && y >= 0 && y < this.viewSize) {
                state.set(1, y, x, 3);
            } else {
                x -= halfViewSize;
                y -= halfViewSize;

                const radius = Math.sqrt(x * x + y * y);

                x = Math.floor(x / radius * halfViewSize) + halfViewSize;
                y = Math.floor(y / radius * halfViewSize) + halfViewSize;

                if (x >= 0 && x < this.viewSize && y >= 0 && y < this.viewSize) {
                    const key = x * 10000 + y;

                    if (!(key in farFoods)) {
                        farFoods[key] = 0;
                    }

                    farFoods[key] += (2 - Math.min(2, (radius - halfViewSize) / halfViewSize)) / 10;
                }
            }
        }

        for (let key in farFoods) {
            const x = Math.floor(key / 10000);
            const y = key % 10000;

            state.set(Math.min(1, farFoods[key]), y, x, 4)
        }

        return state.toTensor();
    }
};