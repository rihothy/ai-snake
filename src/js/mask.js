'use strict';

// L1 + L2 action mask, mirroring train/engine.cpp buildMask exactly.
//
// L1: moves that immediately hit a wall or a body segment are invalid. The
// tail cell is allowed (the tail always vacates this tick; a growth tick only
// duplicates the new tail in place, so the old tail cell stays free).
//
// L2: after the move, the head must still be able to reach the post-move tail
// (the old second-to-last body segment). If not, the snake is guaranteed to
// trap itself eventually, so the move is invalid. Other snakes are treated as
// static bodies minus their tails (same approximation as the engine).

function buildActionMask(snake, snakes, gridWidth, gridHeight) {
    const mask = [0, 0, 0];
    const head = snake.body[0];
    const W = gridWidth, H = gridHeight;

    for (let action = 0; action < 3; ++action) {
        const dir = (snake.direction + action + 3) % 4;
        const nx = head.x + [1, 0, -1, 0][dir];
        const ny = head.y + [0, 1, 0, -1][dir];

        if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;

        let blocked = false;
        for (const other of snakes) {
            if (!other.alive) continue;
            for (let k = 0; k + 1 < other.body.length; ++k) {
                if (other.body[k].x === nx && other.body[k].y === ny) {
                    blocked = true;
                    break;
                }
            }
            if (blocked) break;
        }
        if (blocked) continue;

        if (!canReachTail(snake, snakes, nx, ny, W, H)) continue;
        mask[action] = 1;
    }
    return mask;
}

function canReachTail(snake, snakes, headX, headY, W, H) {
    const body = snake.body;
    const goal = body[body.length - 2];
    const blocked = new Uint8Array(W * H);

    for (let k = 0; k + 2 < body.length; ++k) {
        const s = body[k];
        if (s.x >= 0 && s.x < W && s.y >= 0 && s.y < H) {
            blocked[s.y * W + s.x] = 1;
        }
    }
    for (const other of snakes) {
        if (other === snake || !other.alive) continue;
        for (let k = 0; k + 1 < other.body.length; ++k) {
            const s = other.body[k];
            if (s.x >= 0 && s.x < W && s.y >= 0 && s.y < H) {
                blocked[s.y * W + s.x] = 1;
            }
        }
    }

    const gx = goal.x, gy = goal.y;
    if (gx < 0 || gx >= W || gy < 0 || gy >= H) return false;
    blocked[gy * W + gx] = 0;  // goal is the endpoint
    if (headX === gx && headY === gy) return true;

    const queue = [headY * W + headX];
    blocked[headY * W + headX] = 1;
    const dxs = [1, -1, 0, 0];
    const dys = [0, 0, 1, -1];
    for (let qi = 0; qi < queue.length; ++qi) {
        const cur = queue[qi];
        const cx = cur % W, cy = Math.floor(cur / W);
        for (let d = 0; d < 4; ++d) {
            const nx = cx + dxs[d], ny = cy + dys[d];
            if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;
            const idx = ny * W + nx;
            if (blocked[idx]) continue;
            if (nx === gx && ny === gy) return true;
            blocked[idx] = 1;
            queue.push(idx);
        }
    }
    return false;
}

module.exports = { buildActionMask };
