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

    let lastTimestamp = performance.now();
    let playerDirection = 'right';
    let gameplayInterval = 150;
    let gameplayDeltaTime = 0;

    vars.foodManager = new FoodManager();
    vars.effectManager = new EffectManager();
    vars.agent = await DQNAgent.create('https://raw.githubusercontent.com/rihothy/ai-snake/main/model/model.json');

    for (let i = 0; i < 5; i++) {
        vars.foodManager.generateFood();
    }

    for (let color of cfgs.aiColors) {
        vars.snakes.push(new Snake(color));
    }

    document.addEventListener('keydown', (ev) => {
        if (vars.snakes.length && vars.snakes[0].isPlayer) {
            switch (ev.key) {
                case 'ArrowUp': case 'w': playerDirection = 'up'; break;
                case 'ArrowDown': case 's': playerDirection = 'down'; break;
                case 'ArrowLeft': case 'a': playerDirection = 'left'; break;
                case 'ArrowRight': case'd': playerDirection = 'right'; break;
            }
        }

        if (ev.key == 'Escape') {
            const timeDilation = parseFloat(prompt('change game speed'));

            if (!isNaN(timeDilation) && timeDilation > 0) {
                cfgs.timeDilation = timeDilation;
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

    const tick = (timestamp) => {
        if (timestamp - lastTimestamp > 2000) {
            lastTimestamp = timestamp - gameplayInterval;
        }

        const deltaTime = (timestamp - lastTimestamp) * cfgs.timeDilation;

        gameplayDeltaTime += deltaTime;
        lastTimestamp = timestamp;

        while (gameplayDeltaTime >= gameplayInterval) {
            gameplayDeltaTime -= gameplayInterval;

            tf.tidy(() => {
                for (const snake of vars.snakes) {
                    if (!snake.isPlayer) {
                        snake.move(['up', 'right', 'down', 'left'][vars.agent.getAction(vars.agent.getState(snake), 0)]);
                    } else {
                        snake.move(playerDirection);
                    }
                }
            });

            vars.snakes.forEach(snake => snake.checkCollision());
            vars.snakes.forEach(snake => snake.checkFood());

            vars.foodManager.gameplayTick(gameplayInterval);

            for (let i = vars.snakes.length - 1; i >= 0; i--) {
                if (!vars.snakes[i].states.alive) {
                    for (let j = 0; j < vars.snakes[i].body.length; j++) {
                        const segment = vars.snakes[i].body[j];

                        vars.effectManager.createSnakeEffect(segment.x * cfgs.gridSize + cfgs.gridSize / 2, segment.y * cfgs.gridSize + cfgs.gridSize / 2, vars.snakes[i].color, j % 2 == 0);

                        if (j % 2 == 0) {
                            vars.foodManager.delayGenerateFood(segment.x, segment.y, 1000);
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
        vars.snakes.forEach(snake => {snake.animationProgress = gameplayDeltaTime / gameplayInterval; snake.renderTick(deltaTime);});

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        vars.foodManager.render(ctx);
        vars.snakes.forEach(snake => snake.render(ctx));
        vars.effectManager.render(ctx);

        requestAnimationFrame(tick);
    };

    requestAnimationFrame(tick);
});