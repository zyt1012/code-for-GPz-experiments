from __future__ import annotations

import numpy as np
import tensorflow as tf


class GPzGaussian(tf.Module):
    """
    GPz (Gaussian likelihood) with optional heteroscedastic precision beta(x).

    This matches the Student-t GPz implementation but with IRLS weights fixed to 1.
    Mini-batch objective approximates Gaussian log marginal likelihood (evidence-like).
    """

    def __init__(self, d: int, m: int, seed: int = 0, hetero: bool = True, beta_clip: float = 3.0):
        super().__init__()
        rng = np.random.default_rng(seed)

        self.P = tf.Variable(rng.normal(size=(m, d)).astype(np.float64), name="P")
        self.gamma_raw = tf.Variable(np.zeros((d,), dtype=np.float64), name="gamma_raw")
        self.alpha_raw = tf.Variable(np.zeros((m,), dtype=np.float64), name="alpha_raw")

        self.u = tf.Variable(np.zeros((m, 1), dtype=np.float64), name="u")
        self.b = tf.Variable(np.array(0.0, dtype=np.float64), name="b")

        self.s2_floor_raw = tf.Variable(np.array(-3.0, dtype=np.float64), name="s2_floor_raw")

        self.hetero = bool(hetero)
        self.beta_clip = float(beta_clip)

        self._eps64 = tf.constant(1e-9, dtype=tf.float64)
        self._jitter64 = tf.constant(1e-6, dtype=tf.float64)
        self._one64 = tf.constant(1.0, dtype=tf.float64)
        self._half64 = tf.constant(0.5, dtype=tf.float64)
        self._two_pi64 = tf.constant(2.0 * np.pi, dtype=tf.float64)

    def gamma(self) -> tf.Tensor:
        return tf.nn.softplus(self.gamma_raw) + 1e-6

    def alpha(self) -> tf.Tensor:
        return tf.nn.softplus(self.alpha_raw) + 1e-2  # alpha floor

    def sigma2_floor(self) -> tf.Tensor:
        return tf.nn.softplus(self.s2_floor_raw) + 1e-6

    def phi(self, X: tf.Tensor) -> tf.Tensor:
        diff = X[:, None, :] - self.P[None, :, :]
        g = self.gamma()[None, None, :]
        dist2 = tf.reduce_sum(tf.square(g * diff), axis=2)
        return tf.exp(-self._half64 * dist2)

    def beta_raw(self, Phi: tf.Tensor) -> tf.Tensor:
        if not self.hetero:
            b = tf.clip_by_value(self.b, -self.beta_clip, self.beta_clip)
            beta0 = tf.exp(b)
            n = tf.shape(Phi)[0]
            return beta0 * tf.ones((n, 1), dtype=tf.float64)

        g = tf.linalg.matmul(Phi, self.u) + self.b
        g = tf.clip_by_value(g, -self.beta_clip, self.beta_clip)
        return tf.exp(g)

    def beta_eff(self, beta: tf.Tensor) -> tf.Tensor:
        s2 = self.sigma2_floor()
        return beta / (self._one64 + beta * s2)

    def _posterior_from_precision(self, Phi: tf.Tensor, y: tf.Tensor, precision: tf.Tensor):
        sqrt_p = tf.sqrt(precision)
        Phi_w = Phi * sqrt_p
        y_w = y * sqrt_p

        alpha = self.alpha()
        A = tf.linalg.diag(alpha)

        Sigma = tf.linalg.matmul(Phi_w, Phi_w, transpose_a=True) + A
        Sigma = Sigma + self._jitter64 * tf.eye(tf.shape(Sigma)[0], dtype=tf.float64)
        L = tf.linalg.cholesky(Sigma)

        rhs = tf.linalg.matmul(Phi_w, y_w, transpose_a=True)
        wbar = tf.linalg.cholesky_solve(L, rhs)
        return wbar, L, A

    def log_objective_batch(self, Xb: tf.Tensor, yb: tf.Tensor, n_total: tf.Tensor) -> tf.Tensor:
        """
        Evidence-like Gaussian objective on a mini-batch:
          quad + logdet terms + prior quad + constants - regularisers
        """
        Phi_b = self.phi(Xb)
        beta_b = self.beta_eff(self.beta_raw(Phi_b))  # precision

        bsz = tf.cast(tf.shape(Xb)[0], tf.float64)
        n_total = tf.cast(n_total, tf.float64)
        scale = n_total / tf.maximum(bsz, 1.0)

        wbar, L, A = self._posterior_from_precision(Phi_b, yb, beta_b)
        r = yb - tf.linalg.matmul(Phi_b, wbar)

        quad = -self._half64 * scale * tf.reduce_sum(beta_b * tf.square(r))
        prior_quad = self._half64 * tf.reduce_sum(wbar * tf.linalg.matmul(A, wbar))
        logdetB = self._half64 * scale * tf.reduce_sum(tf.math.log(beta_b + self._eps64))
        logdetA = -self._half64 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(A)))
        logdetSigma = -tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)))
        const = -self._half64 * n_total * tf.math.log(self._two_pi64)

        # regularisation
        reg = tf.constant(1e-4, dtype=tf.float64) * (tf.reduce_sum(tf.square(self.u)) + tf.square(self.b))
        reg += tf.constant(1e-4, dtype=tf.float64) * tf.square(self.s2_floor_raw)

        if self.hetero:
            g_unclipped = tf.linalg.matmul(Phi_b, self.u) + self.b
            reg += tf.constant(1e-2, dtype=tf.float64) * scale * tf.reduce_sum(tf.square(g_unclipped))

        return quad + prior_quad + logdetB + logdetA + logdetSigma + const - reg

    def predict(self, X_train: tf.Tensor, y_train: tf.Tensor, X_test: tf.Tensor, debug: bool = False):
        Phi = self.phi(X_train)
        beta = self.beta_eff(self.beta_raw(Phi))
        wbar, L, _ = self._posterior_from_precision(Phi, y_train, beta)

        Phi_s = self.phi(X_test)
        mean = tf.linalg.matmul(Phi_s, wbar)

        V = tf.linalg.triangular_solve(L, tf.transpose(Phi_s), lower=True)
        nu_feat = tf.reduce_sum(tf.square(V), axis=0, keepdims=True)
        model_var = tf.transpose(nu_feat)

        beta_s = self.beta_eff(self.beta_raw(Phi_s))
        noise_var = self._one64 / (beta_s + self._eps64)

        var = model_var + noise_var

        if debug:
            tf.print("\n[GPz-gauss|test] beta_eff mean:", tf.reduce_mean(beta_s))

        return mean, var


def fit_gpz_gaussian(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    m: int = 256,
    iters: int = 3000,
    lr: float = 0.01,
    seed: int = 0,
    verbose: bool = False,
    hetero: bool | None = None,
    beta_clip: float = 3.0,
    batch_size: int = 1024,
):
    X = tf.convert_to_tensor(X_tr, dtype=tf.float64)
    y = tf.convert_to_tensor(y_tr, dtype=tf.float64)

    n = int(X_tr.shape[0])
    if hetero is None:
        hetero = (n >= 5000)

    m = min(m, n)
    model = GPzGaussian(d=X_tr.shape[1], m=m, seed=seed, hetero=bool(hetero), beta_clip=beta_clip)

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
    n_total = tf.constant(n, dtype=tf.int32)

    @tf.function
    def step():
        Xb, yb = next(it)
        with tf.GradientTape() as tape:
            loss = -model.log_objective_batch(Xb, yb, n_total=n_total)
        vars_ = [model.P, model.gamma_raw, model.alpha_raw, model.u, model.b, model.s2_floor_raw]
        grads = tape.gradient(loss, vars_)
        opt.apply_gradients(zip(grads, vars_))
        return loss

    for _ in range(iters):
        _ = step()

    if verbose:
        tf.print("hetero:", model.hetero, "batch_size:", bs, "beta_clip:", model.beta_clip)

    return model


def predict_gpz_gaussian(model: GPzGaussian, X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, debug: bool = False):
    Xtrain = tf.convert_to_tensor(X_tr, dtype=tf.float64)
    ytrain = tf.convert_to_tensor(y_tr, dtype=tf.float64)
    Xtest = tf.convert_to_tensor(X_te, dtype=tf.float64)
    mean, var = model.predict(Xtrain, ytrain, Xtest, debug=debug)
    return mean.numpy(), var.numpy()
