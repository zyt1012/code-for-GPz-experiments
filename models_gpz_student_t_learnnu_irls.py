from __future__ import annotations

import numpy as np
import tensorflow as tf


class GPzStudentTLearnNuIRLS(tf.Module):
    """
    GPz basis model with heteroscedastic precision beta(x) and Student-t likelihood.
    nu is learnable and enters the IRLS weights:
        lambda_i = (nu+1) / (nu + r_i^2 * beta_eff_i)

    Predictive variance uses the Student-t variance inflation factor nu/(nu-2).

    Notes:
    - nu is constrained to > 2 for finite variance:
        nu = 2 + softplus(nu_raw) + 1e-3
    - Uses alpha floor + noise floor + beta clip for stability.
    """

    def __init__(self, d: int, m: int, seed: int = 0, hetero: bool = True, beta_clip: float = 3.0, nu_init: float = 4.0):
        super().__init__()
        rng = np.random.default_rng(seed)

        self.P = tf.Variable(rng.normal(size=(m, d)).astype(np.float64), name="P")
        self.gamma_raw = tf.Variable(np.zeros((d,), dtype=np.float64), name="gamma_raw")
        self.alpha_raw = tf.Variable(np.zeros((m,), dtype=np.float64), name="alpha_raw")

        self.u = tf.Variable(np.zeros((m, 1), dtype=np.float64), name="u")
        self.b = tf.Variable(np.array(0.0, dtype=np.float64), name="b")

        # noise variance floor: sigma2_floor = softplus(s2_floor_raw) + eps
        self.s2_floor_raw = tf.Variable(np.array(-3.0, dtype=np.float64), name="s2_floor_raw")

        # nu: nu = 2 + softplus(nu_raw) + 1e-3
        nu_raw_init = np.log(np.expm1(max(float(nu_init) - 2.001, 1e-3)))
        self.nu_raw = tf.Variable(np.array(nu_raw_init, dtype=np.float64), name="nu_raw")

        self.hetero = bool(hetero)
        self.beta_clip = float(beta_clip)

        self._eps = tf.constant(1e-9, tf.float64)
        self._jitter = tf.constant(1e-6, tf.float64)
        self._one = tf.constant(1.0, tf.float64)
        self._half = tf.constant(0.5, tf.float64)
        self._two_pi = tf.constant(2.0 * np.pi, tf.float64)

    # --- transforms ---
    def nu(self) -> tf.Tensor:
        return 2.0 + tf.nn.softplus(self.nu_raw) + 1e-3

    def gamma(self) -> tf.Tensor:
        return tf.nn.softplus(self.gamma_raw) + 1e-6

    def alpha(self) -> tf.Tensor:
        return tf.nn.softplus(self.alpha_raw) + 1e-2  # alpha floor

    def sigma2_floor(self) -> tf.Tensor:
        return tf.nn.softplus(self.s2_floor_raw) + 1e-6

    # --- basis ---
    def phi(self, X: tf.Tensor) -> tf.Tensor:
        diff = X[:, None, :] - self.P[None, :, :]
        g = self.gamma()[None, None, :]
        dist2 = tf.reduce_sum(tf.square(g * diff), axis=2)
        return tf.exp(-self._half * dist2)

    # --- heteroscedastic precision ---
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
        return beta / (self._one + beta * s2)

    # --- IRLS weight ---
    def lambda_irls(self, beta_eff: tf.Tensor, r: tf.Tensor) -> tf.Tensor:
        nu = self.nu()
        return (nu + self._one) / (nu + tf.square(r) * beta_eff + self._eps)

    # --- posterior (Gaussian in latent weights) given precision ---
    def posterior_from_precision(self, Phi: tf.Tensor, y: tf.Tensor, prec: tf.Tensor):
        sqrt_p = tf.sqrt(prec)
        Phi_w = Phi * sqrt_p
        y_w = y * sqrt_p

        A = tf.linalg.diag(self.alpha())
        Sigma = tf.linalg.matmul(Phi_w, Phi_w, transpose_a=True) + A
        Sigma += self._jitter * tf.eye(tf.shape(Sigma)[0], dtype=tf.float64)
        L = tf.linalg.cholesky(Sigma)
        rhs = tf.linalg.matmul(Phi_w, y_w, transpose_a=True)
        wbar = tf.linalg.cholesky_solve(L, rhs)
        return wbar, L, A

    def log_objective_batch(self, Xb: tf.Tensor, yb: tf.Tensor, n_total: tf.Tensor) -> tf.Tensor:
        """
        Scaled minibatch objective consistent with GPz evidence form + IRLS weights.
        """
        Phi_b = self.phi(Xb)
        beta_b = self.beta_eff(self.beta_raw(Phi_b))

        # First-pass posterior to get residuals for IRLS weights
        w0, _, _ = self.posterior_from_precision(Phi_b, yb, beta_b)
        r = yb - tf.linalg.matmul(Phi_b, w0)

        lam = self.lambda_irls(beta_b, r)
        prec = beta_b * lam  # effective precision

        bsz = tf.cast(tf.shape(Xb)[0], tf.float64)
        n_total_f = tf.cast(n_total, tf.float64)
        scale = n_total_f / tf.maximum(bsz, 1.0)

        wbar, L, A = self.posterior_from_precision(Phi_b, yb, prec)
        r2 = yb - tf.linalg.matmul(Phi_b, wbar)

        quad = -self._half * scale * tf.reduce_sum(prec * tf.square(r2))
        prior_quad = self._half * tf.reduce_sum(wbar * tf.linalg.matmul(A, wbar))

        logdetB = self._half * scale * tf.reduce_sum(tf.math.log(prec + self._eps))
        logdetA = -self._half * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(A)))
        logdetSigma = -tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)))
        const = -self._half * n_total_f * tf.math.log(self._two_pi)

        # Regularisation: keep modest
        reg = tf.constant(1e-4, tf.float64) * (tf.reduce_sum(tf.square(self.u)) + tf.square(self.b))
        reg += tf.constant(1e-4, tf.float64) * tf.square(self.s2_floor_raw)
        reg += tf.constant(1e-4, tf.float64) * tf.square(self.nu_raw)

        if self.hetero:
            g_unclipped = tf.linalg.matmul(Phi_b, self.u) + self.b
            reg += tf.constant(1e-2, tf.float64) * scale * tf.reduce_sum(tf.square(g_unclipped))

        return quad + prior_quad + logdetB + logdetA + logdetSigma + const - reg

    def predict(self, Xtr: tf.Tensor, ytr: tf.Tensor, Xte: tf.Tensor, debug: bool = False):
        Phi = self.phi(Xtr)
        beta = self.beta_eff(self.beta_raw(Phi))
        wbar, L, _ = self.posterior_from_precision(Phi, ytr, beta)

        Phi_s = self.phi(Xte)
        mean = tf.linalg.matmul(Phi_s, wbar)

        V = tf.linalg.triangular_solve(L, tf.transpose(Phi_s), lower=True)
        model_var = tf.transpose(tf.reduce_sum(tf.square(V), axis=0, keepdims=True))

        beta_s = self.beta_eff(self.beta_raw(Phi_s))
        noise_var = self._one / (beta_s + self._eps)

        var_gauss = model_var + noise_var
        nu = self.nu()
        t_factor = nu / tf.maximum(nu - 2.0, 1e-6)
        var = t_factor * var_gauss

        if debug:
            tf.print("[GPz-t-learnnu-IRLS] nu:", nu, "t_factor:", t_factor)
        return mean, var


def fit_gpz_student_t_learnnu(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    m: int = 256,
    iters: int = 3000,
    lr: float = 0.01,
    seed: int = 0,
    batch_size: int = 1024,
    nu_init: float = 4.0,
    hetero: bool | None = None,
    beta_clip: float = 3.0,
    verbose: bool = False,
):
    X = tf.convert_to_tensor(X_tr, dtype=tf.float64)
    y = tf.convert_to_tensor(y_tr, dtype=tf.float64)

    n = int(X_tr.shape[0])
    if hetero is None:
        hetero = (n >= 5000)

    m = min(int(m), n)
    model = GPzStudentTLearnNuIRLS(d=X_tr.shape[1], m=m, seed=seed, hetero=bool(hetero), beta_clip=beta_clip, nu_init=nu_init)

    # initialise inducing centres from training inputs
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
        vars_ = [model.P, model.gamma_raw, model.alpha_raw, model.u, model.b, model.s2_floor_raw, model.nu_raw]
        grads = tape.gradient(loss, vars_)
        opt.apply_gradients(zip(grads, vars_))
        return loss

    for _ in range(int(iters)):
        _ = step()

    if verbose:
        tf.print("hetero:", model.hetero, "batch_size:", bs, "nu:", model.nu())

    return model


def predict_gpz_student_t_learnnu(model: GPzStudentTLearnNuIRLS, X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, debug: bool = True):
    Xtrain = tf.convert_to_tensor(X_tr, dtype=tf.float64)
    ytrain = tf.convert_to_tensor(y_tr, dtype=tf.float64)
    Xtest = tf.convert_to_tensor(X_te, dtype=tf.float64)
    mean, var = model.predict(Xtrain, ytrain, Xtest, debug=debug)
    return mean.numpy(), var.numpy()
