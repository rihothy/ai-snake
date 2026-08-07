"""Training-side bookkeeping that stays in Python (memory, trajectories).

The game simulation itself lives in the C++ `engine` module; this module only
handles per-snake training metadata and replay-memory interaction.
"""

import random
from collections import deque

import numpy as np
import tensorflow as tf

deadInfo = deque(maxlen=5000)
zeroState = tf.zeros((23, 23, 5), 'float32')
zeroMask = np.zeros(3, np.uint8)


def rememberStageA(snake, state, action, value, mask):
    meta = snake.meta
    # Copy, don't keep views: `state` is a row of the engine's per-tick
    # (N,23,23,5) array. A view pins the whole 6.5MB base buffer alive, so
    # storing views made RSS grow with snake lifetime (~64GB at 10k ticks).
    meta['state'] = np.array(state, copy=True)
    meta['mask'] = np.array(mask, copy=True)
    meta['action'] = action
    meta['value'] = value
    meta.setdefault('trajectory', [])


def rememberStageB(snake, agent):
    meta = snake.meta
    if 'state' not in meta:
        return

    steps = agent.steps
    gamma = agent.gamma

    reward = -0.1
    if snake.ate:
        reward = 1
    if not snake.alive:
        reward = -2.5

    trajectory = meta['trajectory']

    for i in range(-1, -steps, -1):
        if len(trajectory) >= abs(i):
            trajectory[i][4] += (gamma ** abs(i)) * reward

    if len(trajectory) >= steps:
        trajectory[-steps][5] = meta['state']
        trajectory[-steps][6] = meta['mask']
        trajectory[-steps][7] = meta['value']
        trajectory[-steps][8] = False

    #     0,      1,     2,      3,         4,         5,         6,        7,
    # state, mask, action, value, reward, nextState, nextMask, nextValue,
    #    8
    # done
    trajectory.append([meta['state'], meta['mask'], meta['action'],
                       meta['value'], reward, zeroState, zeroMask, 0, True])

    def push(entry):
        (state, mask, action, value, reward,
         nextState, nextMask, nextValue, done) = entry
        agent.memory.tempPush(
            state, action, reward, nextState, mask, nextMask, done,
            (abs(reward + (gamma ** steps) * nextValue - value) + 1e-6) ** agent.memory.alpha
        )

    # Standard per-step experience: the N-step transition completed this tick
    # (its nextState/value were just filled above) enters the replay buffer.
    if len(trajectory) > steps:
        push(trajectory[-steps - 1])
        # Older entries are already in the replay buffer; keep only the last
        # `steps` so per-snake memory is O(steps), not O(snake lifetime).
        del trajectory[:len(trajectory) - steps]

    # On death, also push the trailing transitions that were not pushed yet.
    if not snake.alive:
        for i in range(max(-steps, -len(trajectory)), 0):
            push(trajectory[i])
