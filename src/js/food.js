import { cfgs, vars } from './global.js';

export class Food {
    constructor(x, y) {
        this.size = cfgs.gridSize * 0.65 - 1;
        this.lifeTime = 0;
        this.x = x;
        this.y = y;
    }

    renderTick(deltaTime) {
        if ((this.lifeTime += deltaTime), this.lifeTime < 500) {
            this.size = (cfgs.gridSize * 0.65 - 1) * (1 - Math.pow((1 - Math.max(0, (this.lifeTime / 500))), 2)) * 1.25;
        } else if (this.lifeTime < 750) {
            this.size = (cfgs.gridSize * 0.65 - 1) * (Math.pow((1 - Math.max(0, (this.lifeTime - 500) / 250)), 2) * 0.25 + 1);
        } else {
            this.size = (cfgs.gridSize * 0.65 - 1);
        }
    }

    render(ctx) {
        const x = this.x * cfgs.gridSize + (cfgs.gridSize - this.size) / 2;
        const y = this.y * cfgs.gridSize + (cfgs.gridSize - this.size) / 2;

        ctx.fillStyle = cfgs.foodColor;

        ctx.beginPath();
        ctx.moveTo(x + this.size / 4, y);
        ctx.lineTo(x + this.size - this.size / 4, y);
        ctx.arcTo(x + this.size, y, x + this.size, y + this.size / 4, this.size / 4);
        ctx.lineTo(x + this.size, y + this.size - this.size / 4);
        ctx.arcTo(x + this.size, y + this.size, x + this.size - this.size / 4, y + this.size, this.size / 4);
        ctx.lineTo(x + this.size, y + this.size - this.size / 4);
        ctx.arcTo(x + this.size, y + this.size, x + this.size - this.size / 4, y + this.size, this.size / 4);
        ctx.lineTo(x + this.size - this.size / 4, y + this.size);
        ctx.arcTo(x, y + this.size, x, y + this.size - this.size / 4, this.size / 4);
        ctx.lineTo(x, y + this.size - this.size / 4);
        ctx.arcTo(x, y, x + this.size / 4, y, this.size / 4);
        ctx.closePath();
        ctx.fill();
    }
};

export class FoodManager {
    constructor() {
        this.maxFoodCount = Math.floor(cfgs.gridHeight * cfgs.gridWidth / 50);
        this.minFoodCount = 3;
        this.delayFoods = [];
        this.deltaTime = 0;
        this.foods = [];
    }

    generateFood() {
        const food = new Food(0, 0);

        do {
            food.y = Math.floor(Math.random() * cfgs.gridHeight);
            food.x = Math.floor(Math.random() * cfgs.gridWidth);
        } while (vars.checkPositionOccupied(food.x, food.y));

        this.foods.push(food);
    }

    delayGenerateFood(x, y, delay) {
        if (x >= 0 && x < cfgs.gridWidth && y >= 0 && y < cfgs.gridHeight) {
            this.delayFoods.push({x, y, delay});
        }
    }

    gameplayTick(deltaTime) {
        const foodRatio = (this.foods.length - this.minFoodCount) / (this.maxFoodCount - this.minFoodCount);
        const interval = 250 + (5000 - 250) * foodRatio;

        for (let i = this.delayFoods.length - 1; i >= 0; i--) {
            if ((this.delayFoods[i].delay -= deltaTime) <= 0) {
                if (!vars.checkPositionOccupied(this.delayFoods[i].x, this.delayFoods[i].y)) {
                    this.foods.push(new Food(this.delayFoods[i].x, this.delayFoods[i].y));
                }

                this.delayFoods.splice(i, 1);
            }
        }

        this.deltaTime += deltaTime;

        if (this.foods.length < this.maxFoodCount && this.deltaTime >= interval) {
            this.deltaTime -= interval;
            this.generateFood();
        } else if (this.foods.length < this.minFoodCount) {
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
};