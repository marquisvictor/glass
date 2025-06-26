# glass/core/model.py

import numpy as np
from tqdm import tqdm
from core.support import smooth_source_to_target
from utils.kernels import Kernel
from utils.diagnostic import get_AICc
from utils.search import golden_section

class LocalScorer:
    """
    Lightweight class for local weighted regression used in GLASS.
    Computes predictions, residuals, and the trace of the hat matrix.
    """
    def __init__(self, coords, X, y, bw, kernel='gaussian', fixed=False):
        self.coords = coords
        self.X = X
        self.y = y.reshape(-1, 1)
        self.bw = int(round(bw))
        self.kernel = kernel
        self.fixed = fixed
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
    """
    Generalized Local Additive Spatial Smoothing (GLASS) model class.

    Walkthrough of the Algorithm:
    -----------------------------
    1. Initialize model parameters including bandwidths and smoothing K
    2. For each iteration:
        a. For each covariate:
            i. Apply K smoothing (if needed)
            ii. Exclude the covariate to form X_others
            iii. For each candidate K (or None), search for best bw using AICc
        b. Check if the updates have converged
    3. After convergence, estimate the final model using the optimal K and bw
    """

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
            print(f"\n=== Iteration {iteration + 1} ===")

            var_order = sorted(range(self.p), key=lambda j: 0 if self.support_map[j]['role'] == 'source' else 1)
            print("Variable optimization order:", var_order)

            for j in tqdm(var_order, desc=f"Optimizing variables [1–{self.p}]"):
                print(f"\n  [Start Variable {j}]")
                best_score = np.inf
                best_K = None
                best_bw = None

                K_candidates = self.K_grid if self.support_map[j]['role'] == 'source' else [None]

                for candidate_K in K_candidates:
                    score_log = {}

                    def score_func(bw):
                        bw_rounded = int(round(bw))
                        if bw_rounded in score_log:
                            return score_log[bw_rounded]

                        X_full = []
                        for k in range(self.p):
                            if k == j:
                                xk = self._transform(k, candidate_K)
                            elif self.support_map[k]['role'] == 'source':
                                xk = self._transform(k, self.K[k])
                            else:
                                xk = self.X_raw[k]
                            X_full.append(xk.reshape(-1, 1))
                        X = np.hstack(X_full)

                        model = LocalScorer(self.coords, X, self.y, bw_rounded, fixed=False).fit()
                        aicc = get_AICc(model)
                        score_log[bw_rounded] = aicc
                        print(f"    [Score] Var {j} | K={candidate_K} | bw={bw_rounded} | AICc={aicc:.4f}")
                        return aicc

                    # 1. Optimize using golden section
                    bw_opt, _, _ = golden_section(
                        a=self.bw_range[0],
                        c=self.bw_range[1],
                        delta=0.38197,
                        function=score_func,
                        tol=self.tol_bw,
                        max_iter=100,
                        bw_max=self.bw_range[1],
                        int_score=True,
                        verbose=False
                    )

                    # 2. Check global bw explicitly
                    bw_global = self.n
                    aicc_global = score_func(bw_global)
                    score_opt = score_log[int(round(bw_opt))]

                    if aicc_global < score_opt:
                        print(f"    [Fallback] Global bw={bw_global} has lower AICc ({aicc_global:.4f}) than GSS-opt bw={int(round(bw_opt))} (AICc={score_opt:.4f})")
                        score = aicc_global
                        bw_final = bw_global
                    else:
                        score = score_opt
                        bw_final = int(round(bw_opt))

                    if score < best_score:
                        best_score = score
                        best_K = candidate_K
                        best_bw = bw_final

                self.K[j] = best_K
                self.bw[j] = best_bw
                print(f"  [Final Selection] Var {j} => K={best_K}, bw={best_bw}, AICc={best_score:.4f}")

            if self._has_converged(last_K, last_bw, tol):
                print("\nConvergence achieved.")
                break
            last_K, last_bw = self.K[:], self.bw[:]

        self._final_estimation()


    def _transform(self, j, K):
        """
        Transforms variable j using smoothing if it is from source support.
        """
        role = self.support_map[j]['role']
        x_raw = self.X_raw[j]

        if role == 'source':
            if K is None:
                K = self.K_grid[0]
                print(f"[Warning] Variable {j} has undefined K. Using default K={K}.")
            source_coords = self.support_map[j]['source_coords']
            x_smooth = smooth_source_to_target(x_raw, source_coords, self.coords, K)

            if x_smooth.shape[0] != self.n:
                raise ValueError(
                    f"[Error] Smoothed variable {j} has shape {x_smooth.shape}, "
                    f"expected ({self.n},). Check source-to-target smoothing."
                )
            return x_smooth

        else:
            if x_raw.shape[0] != self.n:
                raise ValueError(
                    f"[Error] Target variable {j} has shape {x_raw.shape}, "
                    f"but expected ({self.n},). Check target alignment."
                )
            return x_raw


    def _build_X_exclude(self, j, current_K_dict):
        """
        Construct the design matrix X excluding variable j.
        Each variable uses either:
        - raw X if it's at target support
        - smoothed X (via K) if it's at source support
        """
        cols = []
        for k in range(self.p):
            if k == j:
                continue
            Kk = current_K_dict[k]
            xk = self._transform(k, Kk)  # Only smooths if source role
            cols.append(xk.reshape(-1, 1))
        return np.hstack(cols) if cols else np.empty((self.n, 0))


    def _has_converged(self, last_K, last_bw, tol):
        for k_new, k_old, bw_new, bw_old in zip(self.K, last_K, self.bw, last_bw):
            if k_new != k_old or (bw_old is None or abs(bw_new - bw_old) > tol):
                return False
        return True

    def _final_estimation(self):
        # Final design matrix with optimized K
        X_final = []
        for j in range(self.p):
            if self.support_map[j]['role'] == 'source':
                xj = self._transform(j, self.K[j])
            else:
                xj = self.X_raw[j]
            X_final.append(xj.reshape(-1, 1))

        X = np.hstack(X_final)

        # Estimate local beta coefficients
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
