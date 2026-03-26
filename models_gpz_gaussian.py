from __future__ import annotations

import numpy as np
import tensorflow as tf

from models_gpz_gaussian_core import GPzGaussianCore


def fit_gpz_gaussian(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    m: int = 256,
    iters: int = 3000,
    lr: float = 0.01,
    seed: int = 0,
    batch_size: int = 1024,
    hetero: bool = True,
    beta_clip: float = 3.0,
):
    X = tf.convert_to_tensor(X_tr, dtype=tf.float64)
    y = tf.convert_to_tensor(y_tr, dtype=tf.float64)

    n = int(X_tr.shape[0])
    m = min(int(m), n)
    model = GPzGaussianCore(d=X_tr.shape[1], m=m, seed=seed, hetero=hetero, beta_clip=beta_clip)

    # init centres from training inputs
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=m, replace=False)
    model.P.assign(tf.convert_to_tensor(X_tr[idx, :], dtype=tf.float64))

    opt = tf.optimizers.Adam(learning_rate=lr)

    bs = int(min(batch_size, n))
    ds = (
        tf.data.Dataset.from_tensor_slices((X, y))
        .shuffle(buffer_size=min(n, 10000), seed=seed, reshuffle_each_iteration=True)
        .repeat()
        .batch(bs, drop_remainder=False)
    )
    it = iter(ds)

    @tf.function
    def step():
        Xb, yb = next(it)
        with tf.GradientTape() as tape:
            loss = -model.log_objective_batch(Xb, yb, n_total=tf.constant(n, tf.int32))
        vars_ = [model.P, model.gamma_raw, model.alpha_raw, model.u, model.b, model.s2_floor_raw]
        grads = tape.gradient(loss, vars_)
        opt.apply_gradients(zip(grads, vars_))
        return loss

    for _ in range(int(iters)):
        _ = step()

    return model


def predict_gpz_gaussian(model: GPzGaussianCore, X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, debug: bool = False):
    Xtrain = tf.convert_to_tensor(X_tr, dtype=tf.float64)
    ytrain = tf.convert_to_tensor(y_tr, dtype=tf.float64)
    Xtest = tf.convert_to_tensor(X_te, dtype=tf.float64)
    mean, var = model.predict(Xtrain, ytrain, Xtest, debug=debug)
    return mean.numpy(), var.numpy()
