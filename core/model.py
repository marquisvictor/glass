# glass/core/model.py

import numpy as np
from core.support import smooth_source_to_target
from utils.kernels import Kernel
from utils.diagnostic import get_AICc
from utils.search import golden_section
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalScorer:
    """
    Lightweight class for local weighted regression used in GLASS.
    Computes predictions, residuals, and the trace of the hat matrix.
    """
    def __init__(self, coords, X, y, bw, kernel='gaussian', fixed=False):
        self.coords = coords              # n x 2 spatial coordinates
        self.X = X                        # n x p design matrix
        self.y = y.reshape(-1, 1)         # response vector
        self.bw = int(round(bw))          # bandwidth for kernel smoothing
        self.kernel = kernel              # kernel function type
        self.fixed = fixed                # whether kernel uses fixed bandwidth
        self.n, self.p = X.shape

    def fit(self):
        """
        Fit a local regression model at each observation using kernel weights,
        accumulate predictions, residuals, and hat matrix trace for AICc.
        """
        predy = np.zeros((self.n, 1))
        resid = np.zeros((self.n, 1))
        tr_S = 0.0

        for i in range(self.n):
            # Compute kernel weights centered at observation i
            kernel = Kernel(i, self.coords, bw=self.bw, fixed=self.fixed, function=self.kernel)
            w = kernel.kernel
            W = np.diag(w)

            # Solve local weighted least squares
            XtW = self.X.T @ W
            XtWX = XtW @ self.X
            XtWy = XtW @ self.y
            try:
                beta = np.linalg.solve(XtWX, XtWy)  # beta = (X'WX)^-1 X'Wy
                y_hat = self.X[i, :] @ beta         # predict y at i
                predy[i] = y_hat
                resid[i] = self.y[i] - y_hat
                S_i = self.X[i, :] @ np.linalg.inv(XtWX) @ XtW[:, i]  # leverage
                tr_S += S_i
            except np.linalg.LinAlgError:
                predy[i] = 0
                resid[i] = self.y[i]
                tr_S += 0

        self.predy = predy
        self.resid = resid
        self.tr_S = tr_S
        return self

class GLASS:
    """
    Generalized Local Additive Spatial Smoothing (GLASS) model class.

    Algorithm Overview:
    -------------------
    1. Initialize each variable's smoothing parameter K and bandwidth bw as None
    2. Iterate through backfitting updates:
        a. For each variable j:
            i. Search over K candidates if it is at source support, else use None
            ii. For each K, define score_func to estimate AICc via LocalScorer
            iii. Use golden section search to find optimal bandwidth minimizing AICc
            iv. Track the best K and bw combination for j
        b. Check convergence across all K and bw
    3. After convergence, compute final coefficient surfaces using the selected K and bw
    """
    def __init__(self, coords, y, support_map, K_grid, bw_range=None, tol_bw=1e-3):
        self.coords = coords                                    # target coordinates (n x 2)
        self.y = y.reshape(-1, 1)                                # response vector
        self.support_map = support_map                          # variable support definitions
        self.K_grid = K_grid                                    # K values for source-smoothing
        self.bw_range = bw_range if bw_range else (20, coords.shape[0])  # bandwidth bounds
        self.tol_bw = tol_bw                                    # tolerance for bw convergence
        self.n = y.shape[0]
        self.p = len(support_map)                               # number of predictors
        self.X_raw = [support_map[j]['data'] for j in range(self.p)]  # raw covariates
        self.K = [None] * self.p                                # smoothing param (K)
        self.bw = [None] * self.p                               # bandwidths
        self.params = np.zeros((self.n, self.p))                # beta surfaces

    def fit(self, max_iter=20, tol=1e-4):
        """
        Run backfitting loop to iteratively optimize each predictor's
        bandwidth and (if needed) change-of-support smoothing.
        """
        last_K = [None] * self.p
        last_bw = [None] * self.p

        for iteration in range(max_iter):
            logger.info(f"Iteration {iteration + 1}")
            for j in range(self.p):
                best_score = np.inf
                best_K = None
                best_bw = None

                # Search over K values for source-supported covariates
                K_candidates = self.K_grid if self.support_map[j]['role'] == 'source' else [None]

                for K in K_candidates:
                    # Transform predictor using current K (if needed)
                    xj = self._transform(j, K)
                    X_others = self._build_X_exclude(j)

                    def score_func(bw):
                        Xj_full = np.column_stack([X_others, xj.reshape(-1, 1)])
                        model = LocalScorer(self.coords, Xj_full, self.y, bw, fixed=False).fit()
                        return get_AICc(model)

                    # Golden section search for optimal bandwidth
                    bw_opt, _, _ = golden_section(
                        a=self.bw_range[0],
                        c=self.bw_range[1],
                        delta=0.38197,
                        function=score_func,
                        tol=self.tol_bw,
                        max_iter=50,
                        bw_max=self.bw_range[1],
                        int_score=True,
                        verbose=False
                    )

                    score = score_func(bw_opt)
                    if score < best_score:
                        best_score = score
                        best_K = K
                        best_bw = int(round(bw_opt))

                self.K[j] = best_K
                self.bw[j] = best_bw
                logger.info(f"Var {j}: best K={best_K}, best bw={best_bw}, AICc={best_score:.3f}")

            if self._has_converged(last_K, last_bw, tol):
                logger.info("Converged.")
                break
            last_K, last_bw = self.K[:], self.bw[:]

        # Estimate final local coefficients using optimal K and bw
        self._final_estimation()

    def _transform(self, j, K):
        """
        Transform predictor j based on support type and smoothing parameter K.
        """
        if self.support_map[j]['role'] == 'source':
            if K is None:
                logger.warning(f"Variable {j} has undefined K. Using K={self.K_grid[0]} as default.")
                K = self.K_grid[0]
            source_coords = self.support_map[j]['source_coords']
            return smooth_source_to_target(self.X_raw[j], source_coords, self.coords, K)
        else:
            return self.X_raw[j]

    def _build_X_exclude(self, j):
        """
        Build design matrix excluding the j-th predictor, transformed with K if needed.
        """
        cols = []
        for k in range(self.p):
            if k != j:
                xk = self._transform(k, self.K[k])
                cols.append(xk.reshape(-1, 1))
        return np.hstack(cols) if cols else np.empty((self.n, 0))

    def _has_converged(self, last_K, last_bw, tol):
        """
        Check for convergence in both K and bandwidth values.
        """
        for k_new, k_old, bw_new, bw_old in zip(self.K, last_K, self.bw, last_bw):
            if k_new != k_old or (bw_old is None or abs(bw_new - bw_old) > tol):
                return False
        return True

    def _final_estimation(self):
        """
        Use final optimized K and bandwidth to estimate local coefficients.
        """
        X_final = [self._transform(j, self.K[j]).reshape(-1, 1) for j in range(self.p)]
        X = np.hstack(X_final)

        for i in range(self.n):
            for j in range(self.p):
                xj = X[:, j].reshape(-1, 1)
                kernel = Kernel(i, self.coords, bw=self.bw[j], fixed=False, function='gaussian')
                w = kernel.kernel.reshape(-1, 1)
                xw = xj * w
                XtWX = xw.T @ xj
                XtWy = xw.T @ self.y
                try:
                    beta = np.linalg.solve(XtWX, XtWy)
                    self.params[i, j] = beta.item()
                except np.linalg.LinAlgError:
                    self.params[i, j] = np.nan
