import { cfgs, vars } from './global.js';
import { FoodManager } from './food.js';
import * as tf from '@tensorflow/tfjs';
import { DQNAgent } from './agent.js';
import { Snake } from './snake.js';

class SnakeAI {
    constructor(snake) {
        this.snake = snake;

        this.memories = [];
        this.action;
        this.state;
    }

    move() {
        this.state = vars.agent.getState(this.snake);
        this.action = vars.agent.getAction(this.state);
        this.snake.move(['up', 'right', 'down', 'left'][this.action]);
    }

    memory() {
        if (!this.state) {
            return;
        }

        let nextState = vars.agent.getState(this.snake);
        let reward = -0.1;
        let type = 0;

        if (this.snake.states.ate) {
            reward += 2;
            type = 1;
        }

        if (!this.snake.states.alive) {
            reward = -7.5;

            if (this.snake.states.collideWall) {
                type = 2;
            } else if (this.snake.states.collideSelf) {
                type = 3;
            } else if (this.snake.states.collideOtherSnake) {
                type = 4;
            }
        }

        this.memories.push({state: this.state, action: this.action, reward, nextState, done: !this.snake.alive, type});

        if (!this.snake.states.alive) {
            for (let i = 0; i < this.memories.length; i++) {
                const shouldPush = i > this.memories.length - Math.min(Math.max(10, this.snake.states.survivalCount / 10), 50);

                if (shouldPush || Math.random() < 0.1) {
                    vars.agent.memory.push(this.memories[i].state, this.memories[i].action, this.memories[i].reward, this.memories[i].nextState, this.memories[i].done, this.memories[i].type);
                } else {
                    tf.dispose(this.memories[i]);
                }
            }

            this.memories = [];
        }
    }
}

class TimerManager {
    constructor() {
        this.timers = [];
    }

    gameplayTick(deltaTime) {
        for (let i = this.timers.length - 1; i >= 0; i--) {
            if ((this.timers[i].delay -= deltaTime) <= 0) {
                this.timers[i].callback();
                this.timers.splice(i, 1);
            }
        }
    }

    setTimer(callback, delay) {
        this.timers.push({callback, delay});
    }
}


(async () => {
    let isPlaying = false;
    let sampleRate = 1;

    self.onmessage = (ev) => {
        isPlaying = ev.data.isPlaying;
        sampleRate = ev.data.sampleRate;
    };

    const timerManager = new TimerManager();
    const gameplayInterval = 150;
    const aiControllers = [];

    vars.agent = await DQNAgent.create();
    vars.foodManager = new FoodManager();

    for (let i = 0; i < 5; i++) {
        vars.foodManager.generateFood();
    }

    for (let color of cfgs.aiColors) {
        aiControllers.push(new SnakeAI(vars.snakes[vars.snakes.push(new Snake(color)) - 1]));
    }

    for (let iter = 0; ; iter++) {
        return;
        aiControllers.forEach(ai => ai.move());
        vars.snakes.forEach(snake => snake.checkCollision());
        vars.snakes.forEach(snake => snake.checkFood());

        timerManager.gameplayTick(gameplayInterval);
        vars.foodManager.gameplayTick(gameplayInterval);

        for (const ai of aiControllers) {
            ai.memory();
        }

        for (let i = vars.snakes.length - 1; i >= 0; i--) {
            if (!vars.snakes[i].states.alive) {
                for (let j = 0; j < vars.snakes[i].body.length; j += 2) {
                    vars.foodManager.delayGenerateFood(vars.snakes[i].body[j].x, vars.snakes[i].body[j].y, 1000);
                }

                const color = vars.snakes[i].color;

                timerManager.setTimer(() => {aiControllers.push(new SnakeAI(vars.snakes[vars.snakes.push(new Snake(color)) - 1]));}, 3000);
                aiControllers.splice(i, 1);
                vars.snakes.splice(i, 1);
            }
        }

        if (vars.agent.memory.isValid()) {
            await vars.agent.train('indexeddb://ai-snake-model');
        }

        if (isPlaying && iter % sampleRate === 0) {
            self.postMessage({snakes: vars.snakes, foods: vars.foodManager.foods, sampleRate});
        }
    }
})();