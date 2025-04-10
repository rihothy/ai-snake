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

    vars.foodManager = new FoodManager();
    vars.effectManager = new EffectManager();
    vars.agent = await DQNAgent.create('src/model/model.json') || await DQNAgent.create('https://raw.githubusercontent.com/rihothy/ai-snake/main/src/model/model.json');

    tf.tidy(() => {
        for (let i = 0; i < 5; ++i) {
            vars.agent.model.predict(tf.ones([5, vars.agent.viewSize, vars.agent.viewSize, vars.agent.stateSize]));
        }

        const startTime = performance.now();

        for (let i = 0; i < 10; ++i) {
            vars.agent.model.predict(tf.ones([5, vars.agent.viewSize, vars.agent.viewSize, vars.agent.stateSize]));
        }

        cfgs.timeDilation = Math.min(7.5, 100 / ((performance.now() - startTime) / 10));
    });

    for (let i = 0; i < 5; ++i) {
        vars.foodManager.generateFood();
    }

    for (let color of cfgs.aiColors) {
        vars.snakes.push(new Snake(color));
    }

    document.addEventListener('keydown', (ev) => {
        if (vars.snakes.length && vars.snakes[0].isPlayer) {
            switch (ev.key) {
                case 'ArrowRight': case 'd': vars.snakes[0].setDirection(0); break;
                case 'ArrowDown': case 's': vars.snakes[0].setDirection(1); break;
                case 'ArrowLeft': case 'a': vars.snakes[0].setDirection(2); break;
                case 'ArrowUp': case 'w': vars.snakes[0].setDirection(3); break;
            }
        }
    });

    canvas.addEventListener('dblclick', (ev) => {
        const x = Math.floor(Math.max(0, Math.min(cfgs.gridWidth - 1, ev.offsetX / cfgs.gridSize)));
        const y = Math.floor(Math.max(0, Math.min(cfgs.gridHeight - 1, ev.offsetY / cfgs.gridSize)));

        if ((vars.snakes.length == 0 || !vars.snakes[0].isPlayer) && !vars.checkPositionOccupied(x, y)) {
            vars.snakes.unshift(new Snake(cfgs.playerColor, true, x, y));
        }
    });

    let lastTimestamp = performance.now();
    let gameplayDeltaTime = 0;

    const tick = (timestamp) => {
        if (timestamp - lastTimestamp > 1000) {
            lastTimestamp = timestamp - 150;
        }

        const deltaTime = (timestamp - lastTimestamp) * cfgs.timeDilation;

        gameplayDeltaTime += deltaTime;
        lastTimestamp = timestamp;

        while (gameplayDeltaTime >= 150) {
            gameplayDeltaTime -= 150;

            tf.tidy(() => {
                const states = [];

                for (const snake of vars.snakes) {
                    if (!snake.isPlayer) {
                        states.push(vars.agent.getState(snake).arraySync());
                    }
                }

                if (states.length) {
                    const [v, a] = vars.agent.model.predict(tf.tensor4d(states));
                    const q = v.add(a.sub(a.mean(-1, true)));
                    const actions = [...q.argMax(-1).dataSync()];

                    for (const snake of vars.snakes) {
                        if (!snake.isPlayer) {
                            snake.setDirection(actions.shift());
                        }
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

                        vars.effectManager.createSnakeEffect(segment.x * cfgs.gridSize + cfgs.gridSize / 2, segment.y * cfgs.gridSize + cfgs.gridSize / 2, vars.snakes[i].color, j % 2 == 0);

                        if (j % 2 == 0) {
                            vars.foodManager.generateFood(segment.x, segment.y, 6);
                        }
                    }

                    if (!vars.snakes[i].isPlayer) {
                        const color = vars.snakes[i].color;

                        setTimeout(() => vars.snakes.push(new Snake(color)), 3000 / cfgs.timeDilation);
                    }

                    vars.snakes.splice(i, 1);
                }
            }
        }

        vars.foodManager.renderTick(deltaTime);
        vars.effectManager.renderTick(deltaTime);
        vars.snakes.forEach(snake => {snake.animationProgress = gameplayDeltaTime / 150; snake.renderTick(deltaTime);});

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        vars.foodManager.render(ctx);
        vars.snakes.forEach(snake => snake.render(ctx));
        vars.effectManager.render(ctx);

        requestAnimationFrame(tick);
    };

    requestAnimationFrame(tick);
});