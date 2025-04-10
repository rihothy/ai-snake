gridWidth = 32
gridHeight = 32

snakes = []

agent = None
foodManager = None

def checkPositionOccupied(x, y):
    return any(any((tx == x and ty == y) for tx, ty in snake.body) for snake in snakes) or any((tx == x and ty == y) for tx, ty in foodManager.foods)