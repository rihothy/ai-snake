importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js');
importScripts('../src/global.js', '../src/food.js', '../src/effects.js', '../src/agent.js', '../src/snake.js');

class SnakeAI {
    constructor(snake) {
        this.snake = snake;

        this.directionChanged = false;
        this.memories = [];
        this.action;
        this.state;
    }

    act() {
        this.state = agent.getState(this.snake);
        this.action = agent.getAction(this.state);

        this.snake.setDirection(['up', 'right', 'down', 'left'][this.action]);
        this.directionChanged = this.snake.nextDirection != this.snake.direction;
    }

    memory() {
        if (!this.state) {
            return;
        }

        let nextState = agent.getState(this.snake);
        let reward = -0.1;
        let type = 0;

        if (this.snake.ate) {
            reward += 2;
            type = 1;
        }

        if (this.directionChanged) {
            reward -= 0.05;
        }

        if (!this.snake.alive) {
            reward = -10;

            if (this.snake.collideWall) {
                type = 2;
            } else if (this.snake.collideSelf) {
                type = 3;
            } else if (this.snake.collideSnake) {
                type = 4;
            }
        }

        this.memories.push({state: this.state, action: this.action, reward, nextState, done: !this.snake.alive, type});

        if (!this.snake.alive) {
            for (let i = 0; i < this.memories.length; i++) {
                if ((i > this.memories.length - Math.min(Math.max(10, this.snake.survivalCount / 10), 50)) || Math.random() < 0.1) {
                    agent.memory.push(this.memories[i].state, this.memories[i].action, this.memories[i].reward, this.memories[i].nextState, this.memories[i].done, this.memories[i].type);
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
            this.timers[i].delay -= deltaTime;

            if (this.timers[i].delay <= 0) {
                this.timers[i].callback();
                this.timers.splice(i, 1);
            }
        }
    }

    setTimer(callback, delay) {
        this.timers.push({callback, delay});
    }
}

const aiControllers = [];

timerManager = new TimerManager();
foodManager = new FoodManager();
agent = new DQNAgent();

for (let i = 0; i < 5; i++) {
    foodManager.generateFood();
}

for (let color of aiColors) {
    aiControllers.push(new SnakeAI(snakes[snakes.push(new Snake(color)) - 1]));
}

(async () => {
    if (loadModelPath) {
        const model = await tf.loadLayersModel(loadModelPath);

        model.summary();
        agent.model.setWeights(model.getWeights());
        agent.targetModel.setWeights(model.getWeights());
    }

    agent.model.summary();

    let iter = 1;

    for (let jter = 1; ; jter++) {
        for (const ai of aiControllers) {
            ai.act();
        }

        for (const snake of snakes) {
            snake.move();
        }

        for (const snake of snakes) {
            snake.checkCollision(snakes);
        }

        for (const snake of snakes) {
            snake.checkFood();
        }

        timerManager.gameplayTick(150);
        foodManager.gameplayTick(150);

        for (const ai of aiControllers) {
            ai.memory();
        }

        for (let i = snakes.length - 1; i >= 0; i--) {
            if (!snakes[i].alive) {
                for (let j = 0; j < snakes[i].body.length; j += 2) {
                    const segment = snakes[i].body[j];

                    timerManager.setTimer(() => {foodManager.generateFood(segment.x, segment.y);}, 1000);
                }

                const color = snakes[i].color;

                timerManager.setTimer(() => {aiControllers.push(new SnakeAI(snakes[snakes.push(new Snake(color)) - 1]));}, 3000);

                snakes.splice(i, 1);
                aiControllers.splice(i, 1);
            }
        }

        // if (agent.memory.isValid()) {
        if (jter % 500 == 0) {
            await agent.train();

            if (jter % 1000 == 0) {
                agent.targetModel.setWeights(agent.model.getWeights());
            }

            metric(iter++, agent.epsilon, agent.gamma);
        }

        // self.postMessage({snakes, foods: foodManager.foods});
    }
})();