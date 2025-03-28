/** @typedef {import('./effects.js').EffectManager} EffectManager */
/** @typedef {import('./food.js').FoodManager} FoodManager */
/** @typedef {import('./agent.js').DQNAgent} DQNAgent */
/** @typedef {import('./snake.js').Snake} Snake */

export const cfgs = {
    gridSize: 20,
    gridWidth: 32,
    gridHeight: 32,
    timeDilation: 1,

    foodColor: 'rgb(156, 39, 176)',
    playerColor: 'rgb(76, 175, 80)',
    aiColors: ['rgb(244, 67, 54)', 'rgb(33, 150, 243)', 'rgb(255, 193, 7)', 'rgb(0, 188, 212)', 'rgb(233, 30, 99)']
};

export const vars = {
    /** @type {Snake[]} */
    snakes: [],
    /** @type {DQNAgent} */
    agent: null,
    /** @type {FoodManager} */
    foodManager: null,
    /** @type {EffectManager} */
    effectManager: null,
    checkPositionOccupied: (x, y) => {
        return vars.snakes.some(snake => snake.states.alive && snake.body.some(segment => segment.x === x && segment.y === y)) || vars.foodManager.foods.some(food => food.x === x && food.y === y);
    }
};

export const metricsInfo = [];