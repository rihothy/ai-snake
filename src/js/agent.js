import { cfgs, vars } from './global.js';
import * as tf from '@tensorflow/tfjs';
import { buildActionMask } from './mask.js';

export class DQNAgent {
    constructor() {
        this.viewSize = 23;
        this.stateSize = 5;
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

    getMask(snake) {
        return buildActionMask(snake, vars.snakes, cfgs.gridWidth, cfgs.gridHeight);
    }

    getState(snake) {
        const state = tf.buffer([this.viewSize, this.viewSize, this.stateSize]);
        const halfViewSize = Math.floor(this.viewSize / 2);
        const head = snake.body[0];

        const offsetX = halfViewSize - head.x;
        const offsetY = halfViewSize - head.y;

        const setState = (value, x, y, channel) => {
            const dir = snake.direction;

            if (dir == 0) {
                state.set(value, y, x, channel)
            } else if (dir == 1) {
                state.set(value, this.viewSize - x - 1, y, channel);
            } else if (dir == 2) {
                state.set(value, this.viewSize - y - 1, this.viewSize - x - 1, channel);
            } else if (dir == 3) {
                state.set(value, x, this.viewSize - y - 1, channel);
            }
        };

        const buildSnakeState = (body, channel) => {
            for (let i = body.length - 1; i >= 0; --i) {
                const x = body[i].x + offsetX;
                const y = body[i].y + offsetY;
    
                if (x >= 0 && x < this.viewSize && y >= 0 && y < this.viewSize) {
                    setState(1 - 0.75 * i / body.length, x, y, channel)
                }
            }
        };

        buildSnakeState(snake.body, 0);

        for (const otherSnake of vars.snakes) {
            buildSnakeState(otherSnake.body, 1);
        }

        for (let y = 0; y < this.viewSize; ++y) {
            for (let x = 0; x < this.viewSize; ++x) {
                const gridX = x - offsetX;
                const gridY = y - offsetY;

                if (gridX < 0 || gridX >= cfgs.gridWidth || gridY < 0 || gridY >= cfgs.gridHeight) {
                    setState(1, x, y, 2);
                }
            }
        }

        let farFoods = {};

        for (const food of vars.foodManager.foods) {
            let x = food.x + offsetX;
            let y = food.y + offsetY;

            if (x >= 0 && x < this.viewSize && y >= 0 && y < this.viewSize) {
                setState(1, x, y, 3);
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

            setState(Math.min(1, farFoods[key]), x, y, 4)
        }

        return state.toTensor();
    }
};
