class Food {
    constructor(x, y) {
        this.size = gridSize * 0.65 - 1;
        this.progress = 0;
        this.x = x;
        this.y = y;
    }

    renderTick(deltaTime) {
        if ((this.progress += deltaTime), this.progress < 500) {
            this.size = (gridSize * 0.65 - 1) * (1 - Math.pow((1 - Math.max(0, (this.progress / 500))), 2)) * 1.25;
        } else if (this.progress < 750) {
            this.size = (gridSize * 0.65 - 1) * (Math.pow((1 - Math.max(0, (this.progress - 500) / 250)), 2) * 0.25 + 1);
        } else {
            this.size = (gridSize * 0.65 - 1);
        }
    }

    render(ctx) {
        const radius = this.size / 4;
        const offsetX = (gridSize - this.size) / 2;
        const offsetY = (gridSize - this.size) / 2;
        const x = this.x * gridSize;
        const y = this.y * gridSize;

        ctx.fillStyle = foodColor;
        ctx.beginPath();
        ctx.moveTo(x + offsetX + radius, y + offsetY);
        ctx.lineTo(x + offsetX + this.size - radius, y + offsetY);
        ctx.arcTo(x + offsetX + this.size, y + offsetY, x + offsetX + this.size, y + offsetY + radius, radius);
        ctx.lineTo(x + offsetX + this.size, y + offsetY + this.size - radius);
        ctx.arcTo(x + offsetX + this.size, y + offsetY + this.size, x + offsetX + this.size - radius, y + offsetY + this.size, radius);
        ctx.lineTo(x + offsetX + radius, y + offsetY + this.size);
        ctx.arcTo(x + offsetX, y + offsetY + this.size, x + offsetX, y + offsetY + this.size - radius, radius);
        ctx.lineTo(x + offsetX, y + offsetY + radius);
        ctx.arcTo(x + offsetX, y + offsetY, x + offsetX + radius, y + offsetY, radius);
        ctx.closePath();
        ctx.fill();
    }
}

class FoodManager {
    constructor() {
        this.maxFoodCount = Math.floor(gridHeight * gridWidth / 50);;
        this.minFoodCount = 3;
        this.deltaTime = 0;
        this.foods = [];
    }

    generateFood(x, y) {
        if (x !== undefined && y !== undefined) {
            if (!checkPositionOccupied(x, y) && x >= 0 && x < gridWidth && y >= 0 && y < gridHeight) {
                this.foods.push(new Food(x, y));
            }
        } else {
            do {
                x = Math.floor(Math.random() * gridWidth);
                y = Math.floor(Math.random() * gridHeight);
            } while (checkPositionOccupied(x, y));

            this.foods.push(new Food(x, y));
        }
    }

    gameplayTick(deltaTime) {
        const foodRatio = (this.foods.length - this.minFoodCount) / (this.maxFoodCount - this.minFoodCount);
        const interval = 250 + (5000 - 250) * foodRatio;

        this.deltaTime += deltaTime;

        while (this.foods.length < this.maxFoodCount && this.deltaTime >= interval) {
            this.deltaTime -= interval;
            this.generateFood();
        }

        while (this.foods.length < this.minFoodCount) {
            this.generateFood();
            this.deltaTime = 0;
        }
    }

    renderTick(deltaTime) {
        this.foods.forEach(food => food.renderTick(deltaTime));
    }

    render(ctx) {
        this.foods.forEach(food => food.render(ctx));
    }
}