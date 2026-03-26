from __future__ import annotations

import numpy as np
import tensorflow as tf
import gpflow


def make_kernel(d: int) -> gpflow.kernels.Kernel:
    # Solid baseline kernel for tabular: RBF with ARD lengthscales
    return gpflow.kernels.SquaredExponential(lengthscales=np.ones(d))


def fit_full_gpr(X_tr: np.ndarray, y_tr: np.ndarray):
    """
    Baseline full GP regression (exact-ish) with GPR.
    Complexity ~ O(n^3) so only use on small/medium datasets (Concrete/hour).
    """
    X = tf.convert_to_tensor(X_tr, dtype=tf.float64)
    y = tf.convert_to_tensor(y_tr, dtype=tf.float64)

    kernel = make_kernel(X_tr.shape[1])
    model = gpflow.models.GPR(data=(X, y), kernel=kernel, mean_function=None)
    model.likelihood.variance.assign(1.0)
    gpflow.set_trainable(model.likelihood.variance, False)

    opt = gpflow.optimizers.Scipy()
    opt.minimize(model.training_loss, variables=model.trainable_variables, options=dict(maxiter=200))
    return model


def fit_svgp(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    inducing_m: int,
    iters: int,
    batch_size: int,
    lr: float,
    seed: int = 0,
):
    """
    Sparse variational GP (SVGP) trained by minimizing negative ELBO.
    Supports mini-batches -> scalable to large n (NYC).
    """
    rng = np.random.default_rng(seed)
    X = tf.convert_to_tensor(X_tr, dtype=tf.float64)
    y = tf.convert_to_tensor(y_tr, dtype=tf.float64)
    n, d = X_tr.shape

    m = min(inducing_m, n)
    idx = rng.choice(n, size=m, replace=False)
    Z = tf.convert_to_tensor(X_tr[idx, :], dtype=tf.float64)

    kernel = make_kernel(d)
    likelihood = gpflow.likelihoods.Gaussian(variance=5.0)

    model = gpflow.models.SVGP(
        kernel=kernel,
        likelihood=likelihood,
        inducing_variable=Z,
        num_latent_gps=1,
        whiten=True,
        q_diag=True,
    )

    optimizer = tf.optimizers.Adam(learning_rate=lr)

    ds = tf.data.Dataset.from_tensor_slices((X, y)).shuffle(buffer_size=min(n, 10000), seed=seed).repeat().batch(batch_size)
    it = iter(ds)

    @tf.function
    def step():
        Xb, yb = next(it)
        with tf.GradientTape() as tape:
            loss = model.training_loss((Xb, yb))
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    for _ in range(iters):
        _ = step()

    return model


def predict_gpflow(model, X: np.ndarray):
    Xtf = tf.convert_to_tensor(X, dtype=tf.float64)
    mean, var = model.predict_y(Xtf)
    return mean.numpy(), var.numpy()
