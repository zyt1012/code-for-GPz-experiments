from __future__ import annotations

import numpy as np
import tensorflow as tf


class GPzStudentT(tf.Module):
    """
    GPz-inspired Bayesian basis function model with heteroscedastic precision beta(x),
    trained with a Student-t likelihood via an IRLS / scale-mixture (Gamma) weighting.

    Key idea:
      Student-t can be written as a Gaussian scale-mixture:
        y_i | f_i, λ_i ~ N(f_i, (λ_i * beta_i)^-1)
        λ_i ~ Gamma(ν/2, ν/2)
      E[λ_i | r_i] = (ν+1) / (ν + r_i^2 * beta_i)

    We use per-mini-batch IRLS:
      - compute current posterior mean weights wbar under beta_i
      - compute residuals r_i
      - compute λ_i
      - re-compute objective using effective precision beta_i * λ_i

    This robustifies training and helps calibration on heavy-tailed/outlier-prone data (e.g., NYC).
    """

    def __init__(self, d: int, m: int, seed: int = 0, hetero: bool = True, beta_clip: float = 3.0, nu: float = 4.0):
        super().__init__()
        rng = np.random.default_rng(seed)

        # Basis centers P: [m, d]
        self.P = tf.Variable(rng.normal(size=(m, d)).astype(np.float64), name="P")

        # Global diagonal precision params gamma_k > 0 (softplus)
        self.gamma_raw = tf.Variable(np.zeros((d,), dtype=np.float64), name="gamma_raw")

        # Prior precision alpha_j > 0 (softplus). alpha floor prevents posterior over-confidence.
        self.alpha_raw = tf.Variable(np.zeros((m,), dtype=np.float64), name="alpha_raw")

        # Noise precision head: g(x)=Phi u + b, beta_raw=exp(clip(g))
        self.u = tf.Variable(np.zeros((m, 1), dtype=np.float64), name="u")
        self.b = tf.Variable(np.array(0.0, dtype=np.float64), name="b")

        # Scalar additive noise variance floor (softplus). Helps avoid variance collapse.
        self.s2_floor_raw = tf.Variable(np.array(-3.0, dtype=np.float64), name="s2_floor_raw")

        self.hetero = bool(hetero)
        self.beta_clip = float(beta_clip)
        self.nu = tf.constant(float(nu), dtype=tf.float64)

        # float64 constants
        self._eps64 = tf.constant(1e-9, dtype=tf.float64)
        self._jitter64 = tf.constant(1e-6, dtype=tf.float64)
        self._one64 = tf.constant(1.0, dtype=tf.float64)
        self._half64 = tf.constant(0.5, dtype=tf.float64)
        self._two_pi64 = tf.constant(2.0 * np.pi, dtype=tf.float64)

    def gamma(self) -> tf.Tensor:
        return tf.nn.softplus(self.gamma_raw) + 1e-6  # [d]

    def alpha(self) -> tf.Tensor:
        return tf.nn.softplus(self.alpha_raw) + 1e-2  # [m] alpha floor

    def sigma2_floor(self) -> tf.Tensor:
        return tf.nn.softplus(self.s2_floor_raw) + 1e-6  # scalar

    def phi(self, X: tf.Tensor) -> tf.Tensor:
        diff = X[:, None, :] - self.P[None, :, :]            # [n,m,d]
        g = self.gamma()[None, None, :]                      # [1,1,d]
        dist2 = tf.reduce_sum(tf.square(g * diff), axis=2)   # [n,m]
        return tf.exp(-self._half64 * dist2)

    def beta_raw(self, Phi: tf.Tensor) -> tf.Tensor:
        if not self.hetero:
            b = tf.clip_by_value(self.b, -self.beta_clip, self.beta_clip)
            beta0 = tf.exp(b)
            n = tf.shape(Phi)[0]
            return beta0 * tf.ones((n, 1), dtype=tf.float64)

        g = tf.linalg.matmul(Phi, self.u) + self.b  # [n,1]
        g = tf.clip_by_value(g, -self.beta_clip, self.beta_clip)
        return tf.exp(g)

    def beta_eff(self, beta: tf.Tensor) -> tf.Tensor:
        """Effective precision after adding scalar noise variance floor."""
        s2 = self.sigma2_floor()
        return beta / (self._one64 + beta * s2)

    def lambda_irls(self, beta_eff: tf.Tensor, r: tf.Tensor) -> tf.Tensor:
        """
        E[λ | r] for Student-t scale-mixture:
          λ = (ν+1) / (ν + r^2 * beta_eff)
        """
        nu = self.nu
        r2_beta = tf.square(r) * beta_eff
        return (nu + self._one64) / (nu + r2_beta + self._eps64)

    def _posterior_from_precision(self, Phi: tf.Tensor, y: tf.Tensor, precision: tf.Tensor):
        """Compute wbar and Cholesky of Sigma given per-point precision."""
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

    # ---------- Mini-batch robust evidence (scalable) ----------
    def log_objective_batch(self, Xb: tf.Tensor, yb: tf.Tensor, n_total: tf.Tensor) -> tf.Tensor:
        """
        Robust mini-batch objective: approximate Student-t marginal log-likelihood
        using IRLS weights (lambda) inside a Gaussian-evidence-like objective.

        Per-step cost: O(b m^2).
        """
        Phi_b = self.phi(Xb)                      # [b,m]
        beta_b = self.beta_eff(self.beta_raw(Phi_b))  # [b,1]

        # First pass posterior under beta_b (Gaussian)
        wbar0, _, _ = self._posterior_from_precision(Phi_b, yb, beta_b)
        r = yb - tf.linalg.matmul(Phi_b, wbar0)        # [b,1]

        # IRLS weights for Student-t
        lam = self.lambda_irls(beta_b, r)              # [b,1]
        prec = beta_b * lam                            # [b,1] effective precision

        # Scale factor for stochastic approximation
        bsz = tf.cast(tf.shape(Xb)[0], tf.float64)
        n_total = tf.cast(n_total, tf.float64)
        scale = n_total / tf.maximum(bsz, 1.0)

        # Posterior with effective precision (approx)
        wbar, L, A = self._posterior_from_precision(Phi_b, yb, tf.sqrt(scale) * prec / tf.sqrt(scale))
        # Note: we incorporate scale in the objective terms below (not inside posterior) to keep stable.

        # Residuals under updated posterior
        r2 = yb - tf.linalg.matmul(Phi_b, wbar)

        # Evidence-like terms with effective precision
        quad = -self._half64 * scale * tf.reduce_sum(prec * tf.square(r2))
        prior_quad = self._half64 * tf.reduce_sum(wbar * tf.linalg.matmul(A, wbar))
        logdetB = self._half64 * scale * tf.reduce_sum(tf.math.log(prec + self._eps64))
        logdetA = -self._half64 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(A)))
        logdetSigma = -tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)))

        const = -self._half64 * n_total * tf.math.log(self._two_pi64)

        # Regularisation
        lam_w = tf.constant(1e-4, dtype=tf.float64)
        reg = lam_w * (tf.reduce_sum(tf.square(self.u)) + tf.square(self.b))
        reg = reg + tf.constant(1e-4, dtype=tf.float64) * tf.square(self.s2_floor_raw)

        reg_g = tf.constant(0.0, dtype=tf.float64)
        if self.hetero:
            g_unclipped = tf.linalg.matmul(Phi_b, self.u) + self.b
            lam_g = tf.constant(1e-2, dtype=tf.float64)
            reg_g = lam_g * scale * tf.reduce_sum(tf.square(g_unclipped))

        return quad + prior_quad + logdetB + logdetA + logdetSigma + const - reg - reg_g

    # ---------- Prediction ----------
    def predict(self, X_train: tf.Tensor, y_train: tf.Tensor, X_test: tf.Tensor, debug: bool = False):
        Phi = self.phi(X_train)
        beta = self.beta_eff(self.beta_raw(Phi))
        wbar, L, _ = self._posterior_from_precision(Phi, y_train, beta)

        Phi_s = self.phi(X_test)
        mean = tf.linalg.matmul(Phi_s, wbar)

        V = tf.linalg.triangular_solve(L, tf.transpose(Phi_s), lower=True)  # [m,n*]
        nu_feat = tf.reduce_sum(tf.square(V), axis=0, keepdims=True)        # [1,n*]
        model_var = tf.transpose(nu_feat)                                   # [n*,1]

        beta_s = self.beta_eff(self.beta_raw(Phi_s))
        noise_var = self._one64 / (beta_s + self._eps64)

        # Student-t marginal variance factor: nu/(nu-2) for nu>2
        nu = self.nu
        var_gauss = model_var + noise_var
        t_factor = nu / tf.maximum(nu - 2.0, 1e-6)
        var = t_factor * var_gauss

        if debug:
            tf.print("\n[GPz-t|test] beta_eff stats:",
                     "min=", tf.reduce_min(beta_s),
                     "mean=", tf.reduce_mean(beta_s),
                     "max=", tf.reduce_max(beta_s))
            tf.print("[GPz-t|test] s2_floor (scalar):", self.sigma2_floor())
            tf.print("[GPz-t|test] nu:", self.nu, "t_factor:", t_factor)
            tf.print("[GPz-t|test] model_var stats:",
                     "min=", tf.reduce_min(model_var),
                     "mean=", tf.reduce_mean(model_var),
                     "max=", tf.reduce_max(model_var))
            tf.print("[GPz-t|test] noise_var stats:",
                     "min=", tf.reduce_min(noise_var),
                     "mean=", tf.reduce_mean(noise_var),
                     "max=", tf.reduce_max(noise_var))
            tf.print("[GPz-t|test] total_var stats:",
                     "min=", tf.reduce_min(var),
                     "mean=", tf.reduce_mean(var),
                     "max=", tf.reduce_max(var))

        return mean, var


def fit_gpz_student_t(
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
    nu: float = 4.0,
):
    X = tf.convert_to_tensor(X_tr, dtype=tf.float64)
    y = tf.convert_to_tensor(y_tr, dtype=tf.float64)

    n = int(X_tr.shape[0])
    if hetero is None:
        hetero = (n >= 5000)

    m = min(m, n)
    model = GPzStudentT(d=X_tr.shape[1], m=m, seed=seed, hetero=bool(hetero), beta_clip=beta_clip, nu=nu)

    # Initialize basis centers from random subset of data
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
        Xb, yb = next(it)
        Phi = model.phi(Xb)
        beta = model.beta_eff(model.beta_raw(Phi))
        tf.print("hetero:", model.hetero, "beta_clip:", model.beta_clip, "batch_size:", bs, "nu:", model.nu)
        tf.print("beta_eff stats (batch):", tf.reduce_min(beta), tf.reduce_mean(beta), tf.reduce_max(beta))
        tf.print("s2_floor (scalar):", model.sigma2_floor())

    return model


def predict_gpz_student_t(model: GPzStudentT, X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, debug: bool = True):
    Xtrain = tf.convert_to_tensor(X_tr, dtype=tf.float64)
    ytrain = tf.convert_to_tensor(y_tr, dtype=tf.float64)
    Xtest = tf.convert_to_tensor(X_te, dtype=tf.float64)
    mean, var = model.predict(Xtrain, ytrain, Xtest, debug=debug)
    return mean.numpy(), var.numpy()
