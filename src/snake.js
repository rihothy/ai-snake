const deadReasons = [];
const deadLengths = [];

class Snake {
    constructor(color, isPlayer = false, x = undefined, y = undefined) {
        while (!isPlayer) {
            x = Math.floor(Math.random() * gridWidth);
            y = Math.floor(Math.random() * gridHeight);

            if (!checkPositionOccupied(x, y)) {
                break;
            }
        }

        this.direction = this.nextDirection = 'right';
        this.body = [{x, y}, {x, y}, {x, y}];
        this.collideSnake = false;
        this.collideSelf = false;
        this.isPlayer = isPlayer;
        this.collideWall = false;
        this.survivalCount = 0;
        this.color = color;
        this.foodCount = 0;
        this.alive = true;
        this.ate = false;

        this.lastBody = this.body.map(curPos => ({...curPos}));
        this.animationProgress = 0;
        this.targetRotation = 0;
        this.rotation = 0;
    }

    setDirection(newDirection) {
        this.collideSnake = false;
        this.collideSelf = false;
        this.collideWall = false;
        this.survivalCount++;
        this.ate = false;

        if (!(newDirection == 'up' && this.direction == 'down' || newDirection == 'down' && this.direction == 'up' || newDirection == 'left' && this.direction == 'right' || newDirection == 'right' && this.direction == 'left')) {
            switch (newDirection) {
                case 'up': this.targetRotation = -Math.PI / 2; break;
                case 'down': this.targetRotation = Math.PI / 2; break;
                case 'left': this.targetRotation = Math.PI; break;
                case 'right': this.targetRotation = 0; break;
            }

            while (this.targetRotation < this.rotation - Math.PI) {
                this.targetRotation += Math.PI * 2;
            }

            while (this.targetRotation > this.rotation + Math.PI) {
                this.targetRotation -= Math.PI * 2;
            }

            this.nextDirection = newDirection;
        }
    }

    move() {
        const head = {...this.body[0]};

        switch (this.nextDirection) {
            case 'up': head.y--; break;
            case 'down': head.y++; break;
            case 'left': head.x--; break;
            case 'right': head.x++; break;
        }

        this.lastBody = this.body.map(curPos => ({...curPos}));
        this.direction = this.nextDirection;
        this.body.unshift(head);
        this.body.pop();
    }

    checkCollision() {
        const head = this.body[0];

        while (deadReasons.length > 500) {
            deadReasons.shift();
            deadLengths.shift();
        }

        if (head.x < 0 || head.x >= gridWidth || head.y < 0 || head.y >= gridHeight) {
            deadLengths.push(this.body.length);
            this.collideWall = true;
            this.alive = false;
            deadReasons.push(0);
            return;
        }

        for (let i = 1; i < this.body.length; i++) {
            if (head.x === this.body[i].x && head.y === this.body[i].y) {
                deadLengths.push(this.body.length);
                this.collideSelf = true;
                this.alive = false;
                deadReasons.push(1);
                return;
            }
        }

        for (let snake of snakes) {
            if (snake !== this) {
                for (let segment of snake.body) {
                    if (head.x === segment.x && head.y === segment.y) {
                        deadLengths.push(this.body.length);
                        this.collideSnake = true;
                        this.alive = false;
                        deadReasons.push(2);
                        return;
                    }
                }
            }
        }
    }

    checkFood() {
        for (let i = foodManager.foods.length - 1; this.alive && i >= 0; i--) {
            const food = foodManager.foods[i];

            if (this.body[0].x === food.x && this.body[0].y === food.y) {
                if (effectManager) {
                    effectManager.createFoodEffect(food.x * gridSize + gridSize / 2, food.y * gridSize + gridSize / 2);
                }

                foodManager.foods.splice(i, 1);
                this.foodCount++;
                this.ate = true;
            }
        }

        if (this.ate) {
            let getRequiredFood = () => {
                if (this.body.length < 10) {
                    return 1;
                } else if (this.body.length < 20) {
                    return 2;
                } else if (this.body.length < 30) {
                    return 3;
                } else if (this.body.length < 40) {
                    return 4;
                } else {
                    return 5;
                }
            };

            while (this.foodCount >= getRequiredFood()) {
                const tail = this.body[this.body.length - 1];

                this.foodCount -= getRequiredFood();
                this.lastBody.push({...tail});
                this.body.push({...tail});
            }
        }
    }

    renderTick(deltaTime) {
        const rotationDiff = this.targetRotation - this.rotation;
        const speed = 0.032 * (1 - Math.sqrt(Math.max(0.25, Math.min(Math.PI, Math.abs(rotationDiff)) / Math.PI)));

        if (Math.abs(rotationDiff) > 0.01) {
            this.rotation += rotationDiff * speed * deltaTime;
        } else {
            this.rotation = this.targetRotation;
        }
    }

    render(ctx) {
        const alphaStep = 0.75 / this.body.length;
        const sizeStep = 0.25 / this.body.length;
        
        for (let i = this.body.length - 1; i >= 0; i--) {
            const curPos = this.body[i];
            const lastPos = this.lastBody[i];
            const size = gridSize - 1 - (i * sizeStep * gridSize);
            const alpha = 1 - (i * alphaStep);
            let x = lastPos.x + (curPos.x - lastPos.x) * this.animationProgress;
            let y = lastPos.y + (curPos.y - lastPos.y) * this.animationProgress;

            ctx.fillStyle = this.color.replace('rgb', 'rgba').replace(')', `, ${alpha})`);
            x = x * gridSize + (gridSize - size) / 2;
            y = y * gridSize + (gridSize - size) / 2;

            if (i === 0) {
                ctx.save();
                ctx.translate(x + size/2, y + size/2);
                ctx.rotate(this.rotation);

                ctx.beginPath();
                const radius = size / 4;
                ctx.moveTo(-size/2, -size/2);
                ctx.lineTo(size/2 - radius, -size/2);
                ctx.arcTo(size/2, -size/2, size/2, -size/2 + radius, radius);
                ctx.lineTo(size/2, size/2 - radius);
                ctx.arcTo(size/2, size/2, size/2 - radius, size/2, radius);
                ctx.lineTo(-size/2, size/2);
                ctx.lineTo(-size/2, -size/2);
                ctx.closePath();
                ctx.fill();

                const eyeSize = size / 6;
                const eyeOffset = size / 4;
                ctx.fillStyle = 'white';

                ctx.beginPath();
                ctx.arc(size/2 - eyeOffset, -eyeOffset, eyeSize, 0, Math.PI * 2);
                ctx.fill();

                ctx.beginPath();
                ctx.arc(size/2 - eyeOffset, eyeOffset, eyeSize, 0, Math.PI * 2);
                ctx.fill();

                ctx.fillStyle = 'black';
                const pupilSize = eyeSize / 2;

                ctx.beginPath();
                ctx.arc(size/2 - eyeOffset + pupilSize/2, -eyeOffset, pupilSize, 0, Math.PI * 2);
                ctx.fill();

                ctx.beginPath();
                ctx.arc(size/2 - eyeOffset + pupilSize/2, eyeOffset, pupilSize, 0, Math.PI * 2);
                ctx.fill();

                ctx.restore();
            } else {
                ctx.fillRect(x, y, size, size);
            }
        }
    }
}