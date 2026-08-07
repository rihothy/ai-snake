import tensorflow as tf
import numpy as np
import random
import os
import json
import time as _time

# TensorFlow's default allocator pre-reserves nearly all GPU memory at
# startup, even though this model needs <1GiB. Cap the training pool to 2GiB
# (measured ~4GiB actual with driver/XLA overhead) so the WSL/Windows host
# keeps the rest of the GPU.
_gpus = tf.config.list_physical_devices('GPU')
if _gpus:
    try:
        tf.config.set_logical_device_configuration(
            _gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=2048)],
        )
    except Exception as _gpu_err:
        print('gpu memory limit not applied:', _gpu_err)

import engine
from agent import DQNAgent
from sim import deadInfo, zeroState, rememberStageA, rememberStageB

if __name__ == '__main__':
    writer = tf.summary.create_file_writer('log/')
    games = engine.create_games(128, 32, 64)
    agent = DQNAgent()
    iter = 0

    checkpoint = 'model/model4.h5'
    stateFile = 'model/train_state.json'
    if os.path.exists(checkpoint):
        print('resuming from', checkpoint)
        agent.model.load_weights(checkpoint)
        agent.targetModel.load_weights(checkpoint)
        if os.path.exists(stateFile):
            with open(stateFile) as f:
                trainState = json.load(f)
            iter = int(trainState.get('iter', 0))
            agent.epsilon = float(trainState.get('epsilon', agent.epsilon))
        else:
            agent.epsilon = 0.33
        print(f'resumed at iter {iter}, epsilon {agent.epsilon:.3f}')
    else:
        print('training from scratch (random init)')

    _phases = {}
    _lastMark = _time.perf_counter()
    _tickCount = 0

    def mark(name):
        global _lastMark
        now = _time.perf_counter()
        _phases[name] = _phases.get(name, 0.0) + now - _lastMark
        _lastMark = now

    while True:
        _tickCount += 1
        states, masks = engine.build_states_and_masks(games, agent.viewSize)
        mark('build')

        values = agent.predict(states).numpy()
        mark('predict')

        # Greedy action selection respects the action mask. When every action
        # is masked (the snake is doomed), fall back to the least-bad raw Q.
        maskedValues = np.where(masks == 1, values, -1e9)
        actions = np.argmax(maskedValues, -1)
        allInvalid = masks.sum(axis=1) == 0
        if allInvalid.any():
            actions[allInvalid] = np.argmax(values[allInvalid], -1)
        actions = actions.tolist()

        for i, (game, snake) in enumerate((game, snake) for game in games for snake in game.snakes):
            if random.random() < agent.epsilon:
                validActions = [a for a in range(3) if masks[i, a]]
                action = random.randint(0, 2) if len(validActions) == 0 else random.choice(validActions)
            else:
                action = actions[i]

            rememberStageA(snake, states[i], action, float(values[i, action]), masks[i])
            snake.setDirection(action - 1)
        mark('agent_loop')

        for death in engine.tick(games):
            deadInfo.append(death)
        mark('tick')

        for game in games:
            for snake in game.snakes:
                rememberStageB(snake, agent)
                if not snake.alive:
                    for j in range(0, len(snake.body), 2):
                        game.add_delay_food(snake.body[j][0], snake.body[j][1], 6)
                    game.add_delay_snake(20)
            game.remove_dead_snakes()
        mark('post')

        if len(agent.memory.memories) + agent.memory.newSize >= 75000 and agent.memory.newSize >= 512:
            agent.memory.push(512)
            agent.train()
            iter += 1

            # Gentle exploration restarts: a small epsilon bump on a long
            # interval. Aggressive resets caused policy oscillation.
            epsilonResetInterval = 8000
            epsilonResetValue = 0.15
            if iter % epsilonResetInterval == 0:
                agent.epsilon = max(agent.epsilon, epsilonResetValue)

            with writer.as_default():
                cnts, ates, _, _ = zip(*deadInfo)
                max_cnt, max_ate = max(cnts), max(ates)
                cnts, ates = sum(cnts) / len(cnts), sum(ates) / len(ates)
                print(f'iter: {iter}, epsilon: {agent.epsilon:.2f}, survival: {cnts:.2f}, ate: {ates:.2f}, max survival: {max_cnt}, max ate: {max_ate}')

                tf.summary.scalar('life time', cnts, iter)
                tf.summary.scalar('ate count', ates, iter)

            if iter % 100 == 0:
                agent.model.save('model/model4.h5')
                if iter % 1000 == 0:
                    agent.model.save(f'model/model4_{iter}.h5')
                with open(stateFile, 'w') as f:
                    json.dump({'iter': iter, 'epsilon': agent.epsilon}, f)

        mark('train_end')
        if _tickCount % 100 == 0:
            print('timings ms/tick:', {k: round(v / 100 * 1000, 1) for k, v in _phases.items()},
                  'tick', _tickCount, 'iter', iter)
            _phases.clear()
            _lastMark = _time.perf_counter()
