import { EffectManager } from './effects.js';
import { cfgs, vars } from './global.js';
import { FoodManager } from './food.js';
import * as tf from '@tensorflow/tfjs';
import { DQNAgent } from './agent.js';
import { Snake } from './snake.js';

window.addEventListener('load', async (ev) => {
    const canvas = document.getElementById('gameCanvas');
    const ctx = canvas.getContext('2d');

    cfgs.gridHeight = Math.floor(window.innerHeight / cfgs.gridSize);
    cfgs.gridWidth = Math.floor(window.innerWidth / cfgs.gridSize);
    canvas.height = cfgs.gridHeight * cfgs.gridSize;
    canvas.width = cfgs.gridWidth * cfgs.gridSize;

    cfgs.fastMode = confirm('开启极速模式');
    console.log('fastmode', cfgs.fastMode);

    vars.foodManager = new FoodManager();
    vars.effectManager = cfgs.fastMode ? undefined : new EffectManager();
    vars.agent = await DQNAgent.create('src/model/model.json') || await DQNAgent.create('https://raw.githubusercontent.com/rihothy/ai-snake/main/src/model/model.json');

    for (let i = 0; i < 5; ++i) {
        vars.foodManager.generateFood();
    }

    for (let color of cfgs.aiColors) {
        vars.snakes.push(new Snake(color));
    }

    let lastTimestamp = performance.now();
    let gameplayDeltaTime = 0;

    const tick = (timestamp) => {
        if (timestamp - lastTimestamp > 150) {
            lastTimestamp = timestamp - 150;
        }

        const deltaTime = (timestamp - lastTimestamp) * cfgs.timeDilation;

        gameplayDeltaTime = cfgs.fastMode ? 150 : gameplayDeltaTime + deltaTime;
        lastTimestamp = timestamp;

        while (gameplayDeltaTime >= 150) {
            gameplayDeltaTime -= 150;

            tf.tidy(() => {
                const states = [];

                for (const snake of vars.snakes) {
                    states.push(vars.agent.getState(snake).arraySync());
                }

                if (states.length) {
                    const [v, a] = vars.agent.model.predict(tf.tensor4d(states));
                    const q = v.add(a.sub(a.mean(-1, true)));

                    const actions = [...q.argMax(-1).dataSync()];

                    for (const snake of vars.snakes) {
                        snake.setDirection(actions.shift() - 1);
                    }
                }
            });

            vars.snakes.forEach(snake => snake.move());
            vars.snakes.forEach(snake => snake.checkCollision());
            vars.snakes.forEach(snake => snake.checkFood());

            vars.foodManager.gameplayTick();

            for (let i = vars.snakes.length - 1; i >= 0; i--) {
                if (!vars.snakes[i].alive) {
                    for (let j = 0; j < vars.snakes[i].body.length; ++j) {
                        const segment = vars.snakes[i].body[j];

                        if (vars.effectManager) {
                            vars.effectManager.createSnakeEffect(segment.x * cfgs.gridSize + cfgs.gridSize / 2, segment.y * cfgs.gridSize + cfgs.gridSize / 2, vars.snakes[i].color, j % 2 == 0);
                        }

                        if (j % 2 == 0) {
                            vars.foodManager.generateFood(segment.x, segment.y, 6);
                        }
                    }

                    const color = vars.snakes[i].color;

                    setTimeout(() => vars.snakes.push(new Snake(color)), 3000 / cfgs.timeDilation);
                    vars.snakes.splice(i, 1);
                }
            }
        }

        if (!cfgs.fastMode) {
            vars.foodManager.renderTick(deltaTime);
            vars.effectManager.renderTick(deltaTime);
            vars.snakes.forEach(snake => {snake.animationProgress = gameplayDeltaTime / 150; snake.renderTick(deltaTime);});
        } else {
            vars.snakes.forEach(snake => {snake.animationProgress = 1; snake.rotation = snake.targetRotation;});
        }

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        vars.foodManager.render(ctx);
        vars.snakes.forEach(snake => snake.render(ctx));

        if (vars.effectManager) {
            vars.effectManager.render(ctx);
        }

        requestAnimationFrame(tick);
    };

    requestAnimationFrame(tick);
});