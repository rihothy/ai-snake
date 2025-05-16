import { cfgs, vars } from './global.js';

export class Snake {
    constructor(color, x = undefined, y = undefined) {
        while (x === undefined || y === undefined || vars.checkPositionOccupied(x, y)) {
            y = Math.floor(Math.random() * cfgs.gridHeight);
            x = Math.floor(Math.random() * cfgs.gridWidth);
        }

        this.body = [{x, y}, {x, y}, {x, y}];
        this.color = color;
        this.direction = 0;
        this.foodCount = 0;
        this.alive = true;

        this.lastBody = this.body.map(curPos => ({...curPos}));
        this.animationProgress = 0;
        this.targetRotation = 0;
        this.rotation = 0;
    }

    setDirection(action) {
        this.direction = (this.direction + action + 4) % 4;

        switch (this.direction) {
            case 0: this.targetRotation = 0; break;
            case 1: this.targetRotation = Math.PI / 2; break;
            case 2: this.targetRotation = Math.PI; break;
            case 3: this.targetRotation = -Math.PI / 2; break;
        }

        while (this.targetRotation < this.rotation - Math.PI) {
            this.targetRotation += Math.PI * 2;
        }

        while (this.targetRotation > this.rotation + Math.PI) {
            this.targetRotation -= Math.PI * 2;
        }
    }

    move() {
        const head = {...this.body[0]};

        head.x += [1, 0, -1, 0][this.direction];
        head.y += [0, 1, 0, -1][this.direction];

        this.lastBody = this.body.map(curPos => ({...curPos}));
        this.body.unshift(head);
        this.body.pop();
    }

    checkCollision() {
        const head = this.body[0];

        if (head.x < 0 || head.x >= cfgs.gridWidth || head.y < 0 || head.y >= cfgs.gridHeight) {
            this.alive = false;
            return;
        }

        for (let i = 1; i < this.body.length; ++i) {
            if (head.x === this.body[i].x && head.y === this.body[i].y) {
                this.alive = false;
                return;
            }
        }

        for (let otherSnake of vars.snakes) {
            if (otherSnake !== this) {
                for (let segment of otherSnake.body) {
                    if (head.x === segment.x && head.y === segment.y) {
                        this.alive = false;
                        return;
                    }
                }
            }
        }
    }

    checkFood() {
        for (let i = vars.foodManager.foods.length - 1; this.alive && i >= 0; i--) {
            const food = vars.foodManager.foods[i];

            if (this.body[0].x === food.x && this.body[0].y === food.y) {
                if (vars.effectManager) {
                    vars.effectManager.createFoodEffect(food.x * cfgs.gridSize + cfgs.gridSize / 2, food.y * cfgs.gridSize + cfgs.gridSize / 2);
                }

                vars.foodManager.foods.splice(i, 1);
                ++this.foodCount;
            }
        }

        const getRequiredFoodCount = () => Math.min(5, Math.floor(this.body.length / 10) + 1);

        while (this.alive && this.foodCount >= getRequiredFoodCount()) {
            const tail = this.body[this.body.length - 1];

            this.foodCount -= getRequiredFoodCount();
            this.lastBody.push({...tail});
            this.body.push({...tail});
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