window.addEventListener('load', (ev) => {
    if (runMode !== 'worker') {
        return;
    }

    const canvas = document.getElementById('gameCanvas');
    const worker = new Worker('worker/index.js');
    const ctx = canvas.getContext('2d');

    canvas.width = gridWidth * gridSize;
    canvas.height = gridHeight * gridSize;

    let snakes = [], foods = [];

    let render = () => {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        for (const food of foods) {
            ctx.fillStyle = 'rgb(156, 39, 176)';
            ctx.fillRect(food.x * gridSize, food.y * gridSize, gridSize, gridSize);
        }

        for (const snake of snakes) {
            for (let i = 0; i < snake.body.length; i++) {
                ctx.fillStyle = snake.color.replace('rgb', 'rgba').replace(')', `, ${i ? 0.5 : 1})`);
                ctx.fillRect(snake.body[i].x * gridSize, snake.body[i].y * gridSize, gridSize, gridSize);
            }
        }

        requestAnimationFrame(render);
    }

    worker.onmessage = (ev) => {
        snakes = ev.data.snakes;
        foods = ev.data.foods;
    }

    requestAnimationFrame(render);
});

window.addEventListener('load', async (ev) => {
    if (runMode !== 'client') {
        return;
    }

    const canvas = document.getElementById('gameCanvas');
    const ctx = canvas.getContext('2d');

    gridWidth = Math.floor(window.innerWidth / gridSize);
    gridHeight = Math.floor(window.innerHeight / gridSize);
    canvas.parentElement.style.padding = '0';
    canvas.parentElement.style.boxShadow = '0 0 0 0';

    canvas.width = gridWidth * gridSize;
    canvas.height = gridHeight * gridSize;

    agent = new DQNAgent();
    foodManager = new FoodManager();
    effectManager = new EffectManager();

    for (let i = 0; i < 5; i++) {
        foodManager.generateFood();
    }

    for (let color of aiColors) {
        snakes.push(new Snake(color));
    }

    if (loadModelPath) {
        agent.model = await tf.loadLayersModel(loadModelPath);
    }

    agent.model.summary();
    this.gameplayDeltaTime = 0;

    document.oncontextmenu = () => false;

    document.addEventListener('keydown', (ev) => {
        if (snakes.length && snakes[0].isPlayer) {
            switch (ev.key) {
                case 'ArrowUp': case 'w': snakes[0].setDirection('up'); break;
                case 'ArrowDown': case 's': snakes[0].setDirection('down'); break;
                case 'ArrowLeft': case 'a': snakes[0].setDirection('left'); break;
                case 'ArrowRight': case'd': snakes[0].setDirection('right'); break;
            }
        }
    });

    let touchStartTime = 0;

    document.addEventListener('touchstart', (ev) => {
        touchStartTime = performance.now();
    });

    document.addEventListener('touchend', (ev) => {
        let x = ev.changedTouches[0].clientX;
        let y = ev.changedTouches[0].clientY;

        if (performance.now() - touchStartTime > 500) {
            const speed = parseFloat(prompt('change game speed'));

            if (speed && speed > 0) {
                timeDilation = speed;
            }
        } else if (snakes.length && snakes[0].isPlayer) {
            if (snakes[0].direction == 'right' || snakes[0].direction == 'left') {
                if (y < canvas.height / 2) snakes[0].setDirection('up');
                else snakes[0].setDirection('down');
            } else if (snakes[0].direction == 'up' || snakes[0].direction == 'down') {
                if (x < canvas.width / 2) snakes[0].setDirection('left');
                else snakes[0].setDirection('right');
            }
        } else {
            x = Math.floor(Math.max(0, Math.min(gridWidth - 1, x / gridSize)));
            y = Math.floor(Math.max(0, Math.min(gridHeight - 1, y / gridSize)));

            if ((snakes.length == 0 || !snakes[0].isPlayer) && !checkPositionOccupied(x, y)) {
                snakes.unshift(new Snake(playerColor, true, x, y));
            }
        }
    });

    canvas.addEventListener('dblclick', (ev) => {
        const x = Math.floor(Math.max(0, Math.min(gridWidth - 1, ev.offsetX / gridSize)));
        const y = Math.floor(Math.max(0, Math.min(gridHeight - 1, ev.offsetY / gridSize)));

        if ((snakes.length == 0 || !snakes[0].isPlayer) && !checkPositionOccupied(x, y)) {
            snakes.unshift(new Snake(playerColor, true, x, y));
        }
    });

    let lastTimestamp = performance.now();
    let gameplayDeltaTime = 0;
    let tick = (timestamp) => {
        if (timestamp - lastTimestamp > 2000) {
            lastTimestamp = timestamp - gameplayInterval;
        }

        const deltaTime = (timestamp - lastTimestamp) * timeDilation;

        gameplayDeltaTime += deltaTime;
        lastTimestamp = timestamp;

        while (gameplayDeltaTime >= gameplayInterval) {
            gameplayDeltaTime -= gameplayInterval;
            foodManager.gameplayTick(gameplayInterval);

            tf.tidy(() => {
                const states = [];

                for (const snake of snakes) {
                    if (!snake.isPlayer) {
                        states.push(agent.getState(snake).arraySync());
                    }
                }

                if (states.length) {
                    const actions = [...agent.model.predict(tf.tensor4d(states)).argMax(1).dataSync()];

                    for (const snake of snakes) {
                        if (!snake.isPlayer) {
                            snake.setDirection(['up', 'right', 'down', 'left'][actions.shift()]);
                        }
                    }
                }
            });

            snakes.forEach(snake => snake.move());
            snakes.forEach(snake => snake.checkCollision());
            snakes.forEach(snake => snake.checkFood());

            for (let i = snakes.length - 1; i >= 0; i--) {
                if (!snakes[i].alive) {
                    for (let j = 0; j < snakes[i].body.length; j++) {
                        const segment = snakes[i].body[j];

                        effectManager.createSnakeEffect(segment.x * gridSize + gridSize / 2, segment.y * gridSize + gridSize / 2, snakes[i].color, j % 2 == 0);

                        if (j % 2 == 0) {
                            setTimeout(() => {foodManager.generateFood(segment.x, segment.y);}, 1000 / timeDilation);
                        }
                    }

                    if (!snakes[i].isPlayer) {
                        const color = snakes[i].color;

                        setTimeout(() => snakes.push(new Snake(color)), 3000 / timeDilation);
                    }

                    snakes.splice(i, 1);
                }
            }
        }

        foodManager.renderTick(deltaTime);
        effectManager.renderTick(deltaTime);
        snakes.forEach(snake => {snake.animationProgress = gameplayDeltaTime / gameplayInterval; snake.renderTick(deltaTime);});

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        foodManager.render(ctx);
        snakes.forEach(snake => snake.render(ctx));
        effectManager.render(ctx);

        requestAnimationFrame(tick);
    };

    requestAnimationFrame(tick);
});