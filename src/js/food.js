import { cfgs, vars } from './global.js';

class Food {
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
        ctx.lineTo(x + this.size / 4, y + this.size);
        ctx.arcTo(x, y + this.size, x, y + this.size - this.size / 4, this.size / 4);
        ctx.lineTo(x, y + this.size / 4);
        ctx.arcTo(x, y, x + this.size / 4, y, this.size / 4);
        ctx.closePath();
        ctx.fill();
    }
}

export class FoodManager {
    constructor() {
        this.maxFoodCount = Math.floor(cfgs.gridHeight * cfgs.gridWidth / 50);
        this.minFoodCount = 3;
        this.delayFoods = [];
        this.deltaTime = 0;
        this.foods = [];
    }

    generateFood(x, y, delay) {
        if (delay !== undefined) {
            this.delayFoods.push({x, y, delay});
        } else {
            if (x !== undefined && y !== undefined) {
                if (!vars.checkPositionOccupied(x, y) && x >= 0 && x < cfgs.gridWidth && y >= 0 && y < cfgs.gridHeight) {
                    this.foods.push(new Food(x, y));
                }
            } else {
                do {
                    x = Math.floor(Math.random() * cfgs.gridWidth);
                    y = Math.floor(Math.random() * cfgs.gridHeight);
                } while (vars.checkPositionOccupied(x, y));
    
                this.foods.push(new Food(x, y));
            }
        }
    }

    gameplayTick() {
        const foodRatio = (this.foods.length - this.minFoodCount) / (this.maxFoodCount - this.minFoodCount);
        const interval = 2 + 32 * foodRatio;

        this.deltaTime += 1;

        for (let i = this.delayFoods.length - 1; i >= 0; i--) {
            if (--this.delayFoods[i].delay <= 0) {
                this.generateFood(this.delayFoods[i].x, this.delayFoods[i].y);
                this.delayFoods.splice(i, 1);
            }
        }

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
};