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
    def __init__(self, coords, X, y, bw, kernel='gaussian', fixed=False):
        self.coords = coords
        self.X = X
        self.y = y.reshape(-1, 1)
        self.bw = int(round(bw))
        self.kernel = kernel
        self.fixed = fixed
        self.n, self.p = X.shape

    def fit(self):
        predy = np.zeros((self.n, 1))
        resid = np.zeros((self.n, 1))
        tr_S = 0.0

        for i in range(self.n):
            kernel = Kernel(i, self.coords, bw=self.bw, fixed=self.fixed, function=self.kernel)
            w = kernel.kernel
            W = np.diag(w)
            XtW = self.X.T @ W
            XtWX = XtW @ self.X
            XtWy = XtW @ self.y
            try:
                beta = np.linalg.solve(XtWX, XtWy)
                y_hat = self.X[i, :] @ beta
                predy[i] = y_hat
                resid[i] = self.y[i] - y_hat
                S_i = self.X[i, :] @ np.linalg.inv(XtWX) @ XtW[:, i]
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
    def __init__(self, coords, y, support_map, K_grid, bw_range=None, tol_bw=1e-3):
        self.coords = coords
        self.y = y.reshape(-1, 1)
        self.support_map = support_map
        self.K_grid = K_grid
        self.bw_range = bw_range if bw_range else (20, coords.shape[0])
        self.tol_bw = tol_bw
        self.n = y.shape[0]
        self.p = len(support_map)
        self.X_raw = [support_map[j]['data'] for j in range(self.p)]
        self.K = [None] * self.p
        self.bw = [None] * self.p
        self.params = np.zeros((self.n, self.p))

    def fit(self, max_iter=20, tol=1e-4):
        last_K = [None] * self.p
        last_bw = [None] * self.p

        for iteration in range(max_iter):
            logger.info(f"Iteration {iteration + 1}")
            for j in range(self.p):
                best_score = np.inf
                best_K = None
                best_bw = None
                K_candidates = self.K_grid if self.support_map[j]['role'] == 'source' else [None]

                for K in K_candidates:
                    xj = self._transform(j, K)
                    X_others = self._build_X_exclude(j)
                    def score_func(bw):
                        Xj_full = np.column_stack([X_others, xj.reshape(-1, 1)])
                        model = LocalScorer(self.coords, Xj_full, self.y, bw, fixed=False).fit()
                        return get_AICc(model)

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

        self._final_estimation()

    def _transform(self, j, K):
        if self.support_map[j]['role'] == 'source':
            if K is None:
                logger.warning(f"Variable {j} has undefined K. Using K={self.K_grid[0]} as default.")
                K = self.K_grid[0]
            source_coords = self.support_map[j]['source_coords']
            return smooth_source_to_target(self.X_raw[j], source_coords, self.coords, K)
        else:
            return self.X_raw[j]

    def _build_X_exclude(self, j):
        cols = []
        for k in range(self.p):
            if k != j:
                xk = self._transform(k, self.K[k])
                cols.append(xk.reshape(-1, 1))
        return np.hstack(cols) if cols else np.empty((self.n, 0))

    def _has_converged(self, last_K, last_bw, tol):
        for k_new, k_old, bw_new, bw_old in zip(self.K, last_K, self.bw, last_bw):
            if k_new != k_old or (bw_old is None or abs(bw_new - bw_old) > tol):
                return False
        return True

    def _final_estimation(self):
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
