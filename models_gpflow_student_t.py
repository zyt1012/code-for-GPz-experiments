from __future__ import annotations

import numpy as np

import gpflow
import tensorflow as tf


def fit_svgp_student_t(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    inducing_m: int = 256,
    iters: int = 3000,
    batch_size: int = 1024,
    lr: float = 0.01,
    seed: int = 0,
    nu: float = 4.0,
):
    """
    SVGP baseline with Student-t likelihood (homoscedastic) to isolate robustness effect.
    Uses the same kernel family (RBF) as the Gaussian SVGP baseline (assumed in your models_gpflow.py).

    IMPORTANT:
      - gpflow.likelihoods.StudentT uses parameter 'df' for degrees of freedom in some versions.
        We try both df and nu for compatibility.
    """
    tf.random.set_seed(seed)
    X = tf.convert_to_tensor(X_tr, dtype=tf.float64)
    Y = tf.convert_to_tensor(y_tr.reshape(-1, 1), dtype=tf.float64)

    # Kernel: RBF (same family)
    kernel = gpflow.kernels.SquaredExponential()

    m = min(int(inducing_m), X_tr.shape[0])
    Z = X_tr[np.random.default_rng(seed).choice(X_tr.shape[0], size=m, replace=False)]
    Z = tf.convert_to_tensor(Z, dtype=tf.float64)

    # Likelihood: StudentT
    try:
        likelihood = gpflow.likelihoods.StudentT(df=float(nu))
    except TypeError:
        likelihood = gpflow.likelihoods.StudentT(nu=float(nu))

    model = gpflow.models.SVGP(kernel=kernel, likelihood=likelihood, inducing_variable=Z, num_latent_gps=1)

    opt = tf.optimizers.Adam(learning_rate=lr)

    ds = (
        tf.data.Dataset.from_tensor_slices((X, Y))
        .shuffle(buffer_size=min(X_tr.shape[0], 10000), seed=seed, reshuffle_each_iteration=True)
        .repeat()
        .batch(min(int(batch_size), X_tr.shape[0]))
    )
    it = iter(ds)

    @tf.function
    def step():
        Xb, Yb = next(it)
        with tf.GradientTape() as tape:
            loss = -model.elbo((Xb, Yb))
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    for _ in range(int(iters)):
        _ = step()

    return model


def predict_svgp_student_t(model: gpflow.models.SVGP, X_te: np.ndarray):
    X = tf.convert_to_tensor(X_te, dtype=tf.float64)
    mean, var = model.predict_y(X)  # predictive mean/var of observations
    return mean.numpy(), var.numpy()
