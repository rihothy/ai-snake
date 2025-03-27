const gridSize = 20;
let gridWidth = 32;
let gridHeight = 32;

const gameplayInterval = 150;
const runMode = 'client';
let timeDilation = 1;

const foodColor = 'rgb(156, 39, 176)';
const playerColor = 'rgb(76, 175, 80)';
const aiColors = ['rgb(244, 67, 54)', 'rgb(33, 150, 243)', 'rgb(255, 193, 7)', 'rgb(0, 188, 212)', 'rgb(233, 30, 99)'];

const snakes = [];

let agent;
let foodManager;
let effectManager;

const loadModelPath = 'https://raw.githubusercontent.com/rihothy/ai-snake/main/model/model.json';
const saveModelPath = '';

function checkPositionOccupied(x, y) {
    return snakes.some(snake => snake.alive && snake.body.some(segment => segment.x === x && segment.y === y)) || foodManager.foods.some(food => food.x === x && food.y == y);
}