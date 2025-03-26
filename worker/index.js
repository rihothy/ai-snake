importScripts('../src/global.js', '../src/food.js', '../src/effects.js', '../src/agent.js', '../src/snake.js');
importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js');

class SnakeAI {
    constructor(snake) {
        this.snake = snake;

        this.directionChanged = false;
        this.action;
        this.state;
    }

    act() {
        this.state = agent.getState(this.snake, foodManager.foods, snakes);
        this.action = agent.getAction(this.state);

        this.snake.setDirection(['up', 'right', 'down', 'left'][this.action]);
        this.directionChanged = this.snake.nextDirection != this.snake.direction;
    }

    memory() {
        let nextState = agent.getState(this.snake, foodManager.foods, snakes);
        let reward = -0.1;

        if (this.snake.ate) reward += 1;
        if (!this.snake.alive) reward = -2;
        // if (this.directionChanged) reward -= 0.5

        agent.memory.push(this.state, this.action, reward, nextState, !this.snake.alive);
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

agent = new DQNAgent('indexeddb://snake-ai-model');
// agent = new DQNAgent();
timerManager = new TimerManager();
foodManager = new FoodManager();

for (let i = 0; i < 5; i++) {
    foodManager.generateFood();
}

for (let color of aiColors) {
    aiControllers.push(new SnakeAI(snakes[snakes.push(new Snake(color)) - 1]));
}

(async () => {
    let model = await tf.loadLayersModel('indexeddb://snake-ai-model');
    // let model = null;

    if (model) {
        // agent.epsilon = 0.01;
        agent.model.setWeights(model.getWeights());
        agent.targetModel.setWeights(model.getWeights());
    }

    agent.model.summary();

    for (let iter = 1; ; iter++) {
        timerManager.gameplayTick(150);
        foodManager.gameplayTick(150);

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

        if (iter % 500 == 0) {
            // await agent.train();
            console.log(`[iter: ${iter / 500}] [epsilon: ${agent.epsilon}]`);
        }

        self.postMessage({snakes, foods: foodManager.foods});
    }
})();