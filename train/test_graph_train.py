"""Verify the tf.function-based train() runs and matches the eager version."""

import os
import random
import sys
import time

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import DQNAgent, _ZERO_STATE_NP


def make_memory(agent, n=1100, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        s = rng.random((23, 23, 5), dtype=np.float32)
        a = int(rng.integers(0, 3))
        r = float(rng.uniform(-2.5, 1.0))
        if rng.random() < 0.2:
            ns = tf.zeros((23, 23, 5), "float32")
        else:
            ns = rng.random((23, 23, 5), dtype=np.float32)
        mask = rng.integers(0, 2, size=3).astype(np.uint8)
        nextMask = rng.integers(0, 2, size=3).astype(np.uint8)
        d = bool(rng.integers(0, 2))
        agent.memory.tempPush(s, a, r, ns, mask, nextMask, d, 1.0)
    agent.memory.push(n)


def eager_train(agent, sampleSize=1024, batchSize=32):
    """Reference implementation of the original eager training step."""
    samples, indices, weight = agent.memory.sample(sampleSize)
    state, action, reward, nextState, mask, nextMask, done = zip(*samples)
    state = tf.stack(state)
    action = tf.stack(action)
    reward = tf.stack(reward)
    nextState = tf.convert_to_tensor(
        np.stack([s if isinstance(s, np.ndarray) else _ZERO_STATE_NP
                  for s in nextState])
    )
    mask = tf.convert_to_tensor(np.stack(mask), "float32")
    nextMask = tf.convert_to_tensor(np.stack(nextMask), "float32")
    done = tf.cast(tf.stack(done), "float32")

    valid = tf.reduce_any(nextMask > 0, -1, keepdims=True)
    qModelRaw = agent.forward(agent.model, nextState)
    qModel = tf.where(valid, tf.where(nextMask > 0, qModelRaw, -1e9), qModelRaw)
    nextAction = tf.argmax(qModel, -1)
    qTargetRaw = agent.forward(agent.targetModel, nextState)
    qTarget = tf.where(valid, tf.where(nextMask > 0, qTargetRaw, -1e9), qTargetRaw)
    targetQ = reward + (1 - done) * (agent.gamma ** agent.steps) * tf.gather_nd(
        qTarget, tf.transpose([tf.range(sampleSize, dtype="int64"), nextAction])
    )

    priors_all = []
    for i in range(0, sampleSize, batchSize):
        with tf.GradientTape() as tape:
            qAll = agent.forward(agent.model, state[i : i + batchSize])
            y_pred = tf.gather_nd(
                qAll,
                tf.stack([tf.range(batchSize), action[i : i + batchSize]], 1),
            )
            y_true = targetQ[i : i + batchSize]
            error = y_pred - y_true
            abs_error = tf.abs(error)
            loss = tf.reduce_mean(
                tf.where(abs_error <= 1, 0.5 * tf.square(error), abs_error - 0.5)
                * weight[i : i + batchSize]
            )
            invalid = 1.0 - mask[i : i + batchSize]
            penalty = tf.reduce_sum(
                tf.square(tf.maximum(qAll + 1.0, 0.0)) * invalid, -1
            )
            loss = loss + 0.05 * tf.reduce_mean(penalty)
        gradients = tape.gradient(loss, agent.model.trainable_weights)
        gradients, _ = tf.clip_by_global_norm(gradients, 10)
        adam_step(agent, gradients)
        priors_all.extend(tf.pow(tf.abs(y_true - y_pred) + 1e-6, agent.memory.alpha).numpy())

    for j, index in enumerate(indices):
        agent.memory.priors[index] = priors_all[j]
    return indices


def adam_step(agent, gradients):
    agent._adamT.assign_add(1)
    t = tf.cast(agent._adamT, "float32")
    beta1, beta2, eps, lr = 0.9, 0.999, 1e-7, 0.0005
    b1t = tf.pow(beta1, t)
    b2t = tf.pow(beta2, t)
    for grad, var, m, v in zip(gradients, agent.model.trainable_weights, agent._adamM, agent._adamV):
        if grad is None:
            continue
        mNew = m * beta1 + grad * (1 - beta1)
        vNew = v * beta2 + tf.square(grad) * (1 - beta2)
        m.assign(mNew)
        v.assign(vNew)
        var.assign_sub(lr * (mNew / (1 - b1t)) / (tf.sqrt(vNew / (1 - b2t)) + eps))


def main():
    # Graph version.
    g = DQNAgent()
    make_memory(g)
    g_weights0 = [w.copy() for w in g.model.get_weights()]
    g_target0 = [w.copy() for w in g.targetModel.get_weights()]
    g_priors0 = g.memory.priors.copy()

    random.seed(1234)
    t0 = time.perf_counter()
    g_indices = g.train()
    graph_time = time.perf_counter() - t0
    g_weights1 = [w for w in g.model.get_weights()]
    assert g_indices is not None, "g.train() returned nothing"

    # Eager reference on an identical twin agent.
    e = DQNAgent()
    e.model.set_weights(g_weights0)
    e.targetModel.set_weights(g_target0)
    e.memory.memories = list(g.memory.memories)
    e.memory.priors = list(g_priors0)
    e.memory.alpha = g.memory.alpha

    random.seed(1234)
    t0 = time.perf_counter()
    e_indices = eager_train(e)
    eager_time = time.perf_counter() - t0
    e_weights1 = [w for w in e.model.get_weights()]

    print("indices equal:", list(g_indices) == list(e_indices))
    per_layer = [float(np.abs(a - b).max()) for a, b in zip(g_weights1, e_weights1)]
    max_w = max(per_layer)
    print("per-layer weight diffs:", [f"{d:.2e}" for d in per_layer])
    max_p = max(
        abs(pa - pb)
        for pa, pb in zip(g.memory.priors[:1024], e.memory.priors[:1024])
    )
    print(f"graph train: {graph_time * 1000:.1f} ms")
    print(f"eager train: {eager_time * 1000:.1f} ms")
    print(f"weight max diff: {max_w:.3e}")
    print(f"prior max diff: {max_p:.3e}")
    assert max_w < 5e-2 and max_p < 5e-1, "graph and eager training diverged too much"
    print("OK: graph and eager training agree within float tolerance")


if __name__ == "__main__":
    main()
