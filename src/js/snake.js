import { cfgs, vars, metricsInfo } from './global.js';

export class Snake {
    constructor(color, isPlayer = false, x = undefined, y = undefined) {
        while (!isPlayer && (x === undefined || y === undefined || vars.checkPositionOccupied(x, y))) {
            y = Math.floor(Math.random() * cfgs.gridHeight);
            x = Math.floor(Math.random() * cfgs.gridWidth);
        }

        this.body = [{x, y}, {x, y}, {x, y}];
        this.direction = 'right';
        this.isPlayer = isPlayer;
        this.color = color;

        this.lastBody = this.body.map(curPos => ({...curPos}));
        this.animationProgress = 0;
        this.targetRotation = 0;
        this.rotation = 0;

        this.states = {
            collideOtherSnake: false,
            collideSelf: false,
            collideWall: false,
            foodCount: 0,
            lifeCount: 0,
            alive: true,
            ate: false,
        };
    }

    move(direction) {
        if (!(direction == 'up' && this.direction == 'down' || direction == 'down' && this.direction == 'up' || direction == 'left' && this.direction == 'right' || direction == 'right' && this.direction == 'left')) {
            this.targetRotation = {up: -Math.PI / 2, down: Math.PI / 2, left: Math.PI, right: 0}[direction];
            while (this.targetRotation < this.rotation - Math.PI) this.targetRotation += Math.PI * 2;
            while (this.targetRotation > this.rotation + Math.PI) this.targetRotation -= Math.PI * 2;
            this.direction = direction;
        }

        const dir = {up: [0, -1], down: [0, 1], left: [-1, 0], right: [1, 0]}[this.direction];
        const head = {x: this.body[0].x + dir[0], y: this.body[0].y + dir[1]};

        this.lastBody = this.body.map(curPos => ({...curPos}));
        this.states.lifeCount++;
        this.states.ate = false;
        this.body.unshift(head);
        this.body.pop();
    }

    checkCollision() {
        const head = this.body[0];

        while (metricsInfo.length > 500) {
            metricsInfo.shift();
        }

        if (head.x < 0 || head.x >= cfgs.gridWidth || head.y < 0 || head.y >= cfgs.gridHeight) {
            metricsInfo.push({type: 'collideWall', length: this.body.length, lifeCount: this.states.lifeCount});
            this.states.collideWall = true;
            this.states.alive = false;
        } else {
            for (let i = 1; i < this.body.length; i++) {
                if (head.x === this.body[i].x && head.y === this.body[i].y) {
                    metricsInfo.push({type: 'collideSelf', length: this.body.length, lifeCount: this.states.lifeCount});
                    this.states.collideSelf = true;
                    this.states.alive = false;
                    return;
                }
            }

            for (let otherSnake of vars.snakes) {
                if (otherSnake !== this && otherSnake.states.alive) {
                    for (let segment of otherSnake.body) {
                        if (head.x === segment.x && head.y === segment.y) {
                            metricsInfo.push({type: 'collideOtherSnake', length: this.body.length, lifeCount: this.states.lifeCount});
                            this.states.collideOtherSnake = true;
                            this.states.alive = false;
                            return;
                        }
                    }
                }
            }
        }
    }

    checkFood() {
        for (let i = vars.foodManager.foods.length - 1; this.states.alive && i >= 0; i--) {
            const food = vars.foodManager.foods[i];

            if (this.body[0].x === food.x && this.body[0].y === food.y) {
                if (vars.effectManager) {
                    vars.effectManager.createFoodEffect(food.x * cfgs.gridSize + cfgs.gridSize / 2, food.y * cfgs.gridSize + cfgs.gridSize / 2);
                }

                vars.foodManager.foods.splice(i, 1);
                this.states.foodCount++;
                this.states.ate = true;
            }
        }

        if (this.states.ate) {
            const getRequiredFoodCount = () => Math.min(5, Math.floor(this.body.length / 10) + 1);

            while (this.states.foodCount >= getRequiredFoodCount()) {
                const tail = this.body[this.body.length - 1];

                this.states.foodCount -= getRequiredFoodCount();
                this.lastBody.push({...tail});
                this.body.push({...tail});
            }
        }
    }

    renderTick(deltaTime) {
        const diff = this.targetRotation - this.rotation;
        const speed = 0.032 * (1 - Math.sqrt(Math.min(Math.PI, Math.max(0.25, Math.abs(diff))) / Math.PI));

        this.rotation = Math.abs(diff) > 0.01 ? this.rotation + diff * Math.min(1, speed * deltaTime) : this.targetRotation;
    }

    render(ctx) {
        for (let i = this.body.length - 1; i >= 0; i--) {
            const size = cfgs.gridSize - 1 - (i * 0.25 / this.body.length * cfgs.gridSize);
            const alpha = 1 - (i * 0.75 / this.body.length);

            const x = (this.lastBody[i].x + (this.body[i].x - this.lastBody[i].x) * this.animationProgress) * cfgs.gridSize + (cfgs.gridSize - size) / 2;
            const y = (this.lastBody[i].y + (this.body[i].y - this.lastBody[i].y) * this.animationProgress) * cfgs.gridSize + (cfgs.gridSize - size) / 2;

            if (i) {
                ctx.fillStyle = this.color.replace('rgb', 'rgba').replace(')', `, ${alpha})`);
                ctx.fillRect(x, y, size, size);
            } else {
                ctx.save();
                {
                    ctx.translate(x + size / 2, y + size / 2);
                    ctx.rotate(this.rotation);

                    ctx.fillStyle = this.color;

                    ctx.beginPath();
                    ctx.moveTo(-size / 2, -size / 2);
                    ctx.lineTo(size / 2 - size / 4, -size / 2);
                    ctx.arcTo(size / 2, -size / 2, size / 2, -size / 2 + size / 4, size / 4);
                    ctx.lineTo(size / 2, size / 2 - size / 4);
                    ctx.arcTo(size / 2, size / 2, size / 2 - size / 4, size / 2, size / 4);
                    ctx.lineTo(-size / 2, size / 2);
                    ctx.lineTo(-size / 2, -size / 2);
                    ctx.closePath();
                    ctx.fill();

                    ctx.fillStyle = 'white';

                    ctx.beginPath();
                    ctx.arc(size/2 - size / 4, -size / 4, size / 6, 0, Math.PI * 2);
                    ctx.fill();

                    ctx.beginPath();
                    ctx.arc(size/2 - size / 4, size / 4, size / 6, 0, Math.PI * 2);
                    ctx.fill();

                    ctx.fillStyle = 'black';

                    ctx.beginPath();
                    ctx.arc(size/2 - size / 4 + size / 24, -size / 4, size / 12, 0, Math.PI * 2);
                    ctx.fill();

                    ctx.beginPath();
                    ctx.arc(size/2 - size / 4 + size / 24, size / 4, size / 12, 0, Math.PI * 2);
                    ctx.fill();
                }
                ctx.restore();
            }
        }
    }
};