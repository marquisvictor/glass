#  working script as of June 4th 2025

# glass/core/model.py

import numpy as np
from core.support import smooth_source_to_target
from utils.diagnostic import get_AICc
from utils.search import golden_section
from utils.kernels import Kernel
from tqdm.notebook import tqdm

class LocalScorer:
    """
    Lightweight local regression model used to score candidate bandwidths (bw) and
    smoothing parameters (K) during the backfitting process in GLASS.
    Computes local predictions, residuals, and the trace of the hat matrix.
    """
    def __init__(self, coords, X, y, bw, kernel='gaussian'):
        self.coords = coords              # n x 2 coordinate matrix
        self.X = X                        # n x p design matrix
        self.y = y.reshape(-1, 1)         # n x 1 response vector
        self.bw = bw                      # bandwidth parameter
        self.kernel = kernel              # kernel type (currently fixed to 'gaussian')
        self.n, self.p = X.shape          # number of observations and predictors
        self.predy = np.zeros((self.n, 1))
        self.resid = np.zeros((self.n, 1))
        self.tr_S = 0.0                   # trace of the hat matrix (effective number of parameters)

    def fit(self):
        """ Fit local models for each observation using Gaussian kernel. """
        for i in range(self.n):
            dists = np.sqrt(np.sum((self.coords - self.coords[i]) ** 2, axis=1))
            kernel = Kernel(i, self.coords, bw=self.bw, fixed=True, function=self.kernel)
            w_i = kernel.kernel
            W = np.diag(w_i)
            XtW = self.X.T @ W
            XtWX = XtW @ self.X
            XtWy = XtW @ self.y
            try:
                beta = np.linalg.solve(XtWX, XtWy)
                y_hat = self.X[i, :] @ beta
                self.predy[i] = y_hat
                self.resid[i] = self.y[i] - y_hat
                S_i = self.X[i, :] @ np.linalg.inv(XtWX) @ XtW[:, i]
                self.tr_S += S_i
            except np.linalg.LinAlgError:
                self.predy[i] = 0
                self.resid[i] = self.y[i]
                self.tr_S += 0
        return self

class GLASS:
    """
    Generalized Local Additive Spatial Smoothing (GLASS) model.
    Extends MGWR to support change-of-support via spatial smoothing (parameter K),
    and allows multiscale, multivariate local modeling with internal bandwidth optimization.
    """
    def __init__(self, coords, y, support_map, K_grid, bw_range=None, tol_bw=1e-3):
        self.coords = coords                              # n x 2 coordinates
        self.y = y.reshape(-1, 1)                          # response variable
        self.support_map = support_map                    # list of dictionaries: {'role': 'source'/'target', 'data': array}
        self.K_grid = K_grid                              # candidate K values (for smoothing source to target)
        self.bw_range = bw_range if bw_range else (10, len(self.coords))  # bandwidth search range
        self.tol_bw = tol_bw                              # tolerance for bw optimization
        self.n, self.p = y.shape[0], len(support_map)     # number of obs and predictors
        self.K = [self.K_grid[0] if support_map[j]['role'] == 'source' else None for j in range(self.p)] # K values for each predictor
        self.bw = [None] * self.p                         # bandwidths for each predictor
        self.params = np.zeros((self.n, self.p))          # final estimated beta surfaces
        self.X_raw = [support_map[j]['data'] for j in range(self.p)]

        if y.shape[0] != coords.shape[0]:
            raise ValueError(f"Mismatch between y ({y.shape[0]}) and target support ({coords.shape[0]})")


    def fit(self, max_iter=10, tol=1e-3):
        """
        Fit the GLASS model using iterative backfitting with tqdm progress bar and verbose logging.
        """
        last_K = [None] * self.p
        last_bw = [None] * self.p

        for iteration in tqdm(range(max_iter), desc="GLASS Backfitting Iterations"):
            print(f"\n--- Iteration {iteration + 1} ---")

            for j in range(self.p):
                print(f"\nVariable {j}:")
                best_score = np.inf
                best_K = None
                best_bw = None
                aicc_history = {}

                # Only source-supported variables need K optimization
                K_candidates = self.K_grid if self.support_map[j]['role'] == 'source' else [None]

                for K in K_candidates:
                    xj = self._transform(j, K)
                    X_others = self._build_X_exclude(j)

                    def score_func(bw):
                        Xj_full = np.column_stack([X_others, xj.reshape(-1, 1)])
                        model = LocalScorer(self.coords, Xj_full, self.y, bw)
                        return get_AICc(model.fit())

                    bw_opt, _, _ = golden_section(
                        a=self.bw_range[0],
                        c=self.bw_range[1],
                        delta=0.38197,
                        function=score_func,
                        tol=self.tol_bw,
                        max_iter=50,
                        bw_max=self.bw_range[1],
                        int_score=False,
                        verbose=False
                    )

                    score = score_func(bw_opt)
                    aicc_history[(K, bw_opt)] = score

                    print(f"  K={K}, bw={bw_opt:.2f}, AICc={score:.4f}")

                    if score < best_score:
                        best_score = score
                        best_K = K
                        best_bw = bw_opt

                self.K[j] = best_K
                self.bw[j] = best_bw
                print(f"→ Best for Variable {j}: K = {best_K}, bw = {best_bw:.2f}, AICc = {best_score:.4f}")

            if self._has_converged(last_K, last_bw, tol):
                print("\nModel converged.")
                break

            last_K, last_bw = self.K[:], self.bw[:]

        print("\nBackfitting complete.")

        # Final model estimation with all optimized covariates
        X_final = []
        for j in range(self.p):
            K_j = self.K[j]
            xj = self._transform(j, K_j)
            X_final.append(xj.reshape(-1, 1))
        X = np.hstack(X_final)

        # Local fitting for each variable at each location using its own optimized bandwidth
        for i in range(self.n):
            dists = np.sqrt(np.sum((self.coords - self.coords[i]) ** 2, axis=1))
            for j in range(self.p):
                xj = X[:, j].reshape(-1, 1)
                kernel_obj = Kernel(i, self.coords, bw=self.bw[j], fixed=True, function='gaussian')
                w_j = kernel_obj.kernel.reshape(-1, 1)

                xw = xj * w_j
                XtWX = xw.T @ xj
                XtWy = xw.T @ self.y
                try:
                    beta_j = np.linalg.solve(XtWX, XtWy)
                    self.params[i, j] = beta_j.item()
                except np.linalg.LinAlgError:
                    self.params[i, j] = np.nan

    def _transform(self, j, K):
        if self.support_map[j]['role'] == 'source':
            source_coords = self.support_map[j].get('source_coords')
            if source_coords is None:
                raise ValueError(f"Missing 'source_coords' for source-supported variable {j}")
            x_smooth = smooth_source_to_target(self.X_raw[j], source_coords, self.coords, K)
            if x_smooth.shape[0] != self.n:
                raise ValueError(f"[Transform Error] Variable {j}: Expected smoothed shape ({self.n},), got {x_smooth.shape}")
            return x_smooth
        else:
            return self.X_raw[j]

    def _build_X_exclude(self, j):
        cols = []
        for k in range(self.p):
            if k != j:
                role = self.support_map[k]['role']
                if role == 'source':
                    source_coords = self.support_map[k].get('source_coords')
                    xk = smooth_source_to_target(self.X_raw[k], source_coords, self.coords, self.K[k])
                else:
                    xk = self.X_raw[k]
                if xk.shape[0] != self.n:
                    raise ValueError(f"[Build_X_Exclude Error] Variable {k}: Expected shape ({self.n},), got {xk.shape}")
                cols.append(xk.reshape(-1, 1))
        return np.hstack(cols)

    def _has_converged(self, last_K, last_bw, tol):
        """ Check if both K and bandwidths have stabilized across all variables. """
        for k_new, k_old, bw_new, bw_old in zip(self.K, last_K, self.bw, last_bw):
            if k_new != k_old:
                return False
            if bw_old is None or abs(bw_new - bw_old) > tol:
                return False
        return True


# previous code that I don't know the date of




# glass/core/model.py

import numpy as np
from core.support import smooth_source_to_target
from utils.diagnostics import aicc
from utils.search import golden_section
from utils.kernels import Kernel

class LocalScorer:
    """
    Lightweight local regression model used to score candidate bandwidths (bw) and
    smoothing parameters (K) during the backfitting process in GLASS.
    Computes local predictions, residuals, and the trace of the hat matrix.
    """
    def __init__(self, coords, X, y, bw, kernel='gaussian'):
        self.coords = coords              # n x 2 coordinate matrix
        self.X = X                        # n x p design matrix
        self.y = y.reshape(-1, 1)         # n x 1 response vector
        self.bw = bw                      # bandwidth parameter
        self.kernel = kernel              # kernel type (currently fixed to 'gaussian')
        self.n, self.p = X.shape          # number of observations and predictors
        self.predy = np.zeros((self.n, 1))
        self.resid = np.zeros((self.n, 1))
        self.tr_S = 0.0                   # trace of the hat matrix (effective number of parameters)

    def fit(self):
        """ Fit local models for each observation using Gaussian kernel. """
        for i in range(self.n):
            dists = np.sqrt(np.sum((self.coords - self.coords[i]) ** 2, axis=1))
            w_i = gaussian(dists, self.bw)
            W = np.diag(w_i)
            XtW = self.X.T @ W
            XtWX = XtW @ self.X
            XtWy = XtW @ self.y
            try:
                beta = np.linalg.solve(XtWX, XtWy)
                y_hat = self.X[i, :] @ beta
                self.predy[i] = y_hat
                self.resid[i] = self.y[i] - y_hat
                S_i = self.X[i, :] @ np.linalg.inv(XtWX) @ XtW[:, i]
                self.tr_S += S_i
            except np.linalg.LinAlgError:
                self.predy[i] = 0
                self.resid[i] = self.y[i]
                self.tr_S += 0
        return self

class GLASS:
    """
    Generalized Local Additive Spatial Smoothing (GLASS) model.
    Extends MGWR to support change-of-support via spatial smoothing (parameter K),
    and allows multiscale, multivariate local modeling with internal bandwidth optimization.
    """
    def __init__(self, coords, y, support_map, K_grid, bw_range=None, tol_bw=1e-3):
        self.coords = coords                              # n x 2 coordinates
        self.y = y.reshape(-1, 1)                          # response variable
        self.support_map = support_map                    # list of dictionaries: {'role': 'source'/'target', 'data': array}
        self.K_grid = K_grid                              # candidate K values (for smoothing source to target)
        self.bw_range = bw_range if bw_range else (10, 150)  # bandwidth search range
        self.tol_bw = tol_bw                              # tolerance for bw optimization
        self.n, self.p = y.shape[0], len(support_map)     # number of obs and predictors
        self.K = [None] * self.p                          # K values for each predictor
        self.bw = [None] * self.p                         # bandwidths for each predictor
        self.params = np.zeros((self.n, self.p))          # final estimated beta surfaces
        self.X_raw = [support_map[j]['data'] for j in range(self.p)]

    def fit(self, max_iter=10, tol=1e-3):
        """
        Main model-fitting routine using iterative backfitting to jointly optimize
        bandwidths and change-of-support smoothing parameters (if needed).
        """
        last_K = [None] * self.p
        last_bw = [None] * self.p

        for iteration in range(max_iter):
            for j in range(self.p):
                best_score = np.inf
                best_K = None
                best_bw = None

                # Evaluate K options only for source-supported variables
                K_candidates = self.K_grid if self.support_map[j]['role'] == 'source' else [None]

                for K in K_candidates:
                    xj = self._transform(j, K)
                    X_others = self._build_X_exclude(j)

                    def score_func(bw):
                        Xj_full = np.column_stack([X_others, xj.reshape(-1, 1)])
                        model = LocalScorer(self.coords, Xj_full, self.y, bw)
                        return aicc(model.fit())

                    bw_opt = golden_section(score_func, self.bw_range[0], self.bw_range[1], self.tol_bw)
                    score = score_func(bw_opt)

                    if score < best_score:
                        best_score = score
                        best_K = K
                        best_bw = bw_opt

                self.K[j] = best_K
                self.bw[j] = best_bw

            if self._has_converged(last_K, last_bw, tol):
                break
            last_K, last_bw = self.K[:], self.bw[:]

        # Final model estimation with all optimized covariates
        X_final = []
        for j in range(self.p):
            K_j = self.K[j]
            xj = self._transform(j, K_j)
            X_final.append(xj.reshape(-1, 1))
        X = np.hstack(X_final)

        # Local fitting for each variable at each location using its own optimized bandwidth
        kernel_fn = get_kernel('gaussian')
        for i in range(self.n):
            # Compute distances from focal location to all others
            dists = np.sqrt(np.sum((self.coords - self.coords[i]) ** 2, axis=1))

            for j in range(self.p):
                # Extract j-th covariate as column vector
                xj = X[:, j].reshape(-1, 1)
                # Compute kernel weights for this variable at this location
                w_j = kernel_fn(dists, self.bw[j]).reshape(-1, 1)
                # Apply element-wise weighting instead of building full diagonal matrix
                xw = xj * w_j
                XtWX = xw.T @ xj
                XtWy = xw.T @ self.y
                try:
                    beta_j = np.linalg.solve(XtWX, XtWy)
                    self.params[i, j] = beta_j.item()  # Safe: XtWX and XtWy are 1x1 here
                except np.linalg.LinAlgError:
                    self.params[i, j] = np.nan


    def _transform(self, j, K):
        """ Perform change-of-support smoothing if variable j is at source support. """
        if self.support_map[j]['role'] == 'source' and K is not None:
            return smooth_source_to_target(self.X_raw[j], self.coords, K)
        return self.X_raw[j]

    def _build_X_exclude(self, j):
        """ Build design matrix excluding variable j. Used in backfitting. """
        cols = []
        for k in range(self.p):
            if k != j:
                xk = self._transform(k, self.K[k])
                cols.append(xk.reshape(-1, 1))
        return np.hstack(cols)

    def _has_converged(self, last_K, last_bw, tol):
        """ Check if both K and bandwidths have stabilized across all variables. """
        for k_new, k_old, bw_new, bw_old in zip(self.K, last_K, self.bw, last_bw):
            if k_new != k_old:
                return False
            if bw_old is None or abs(bw_new - bw_old) > tol:
                return False
        return True
    


    ###  older

# glass/core/model.py

import numpy as np
from core.support import smooth_source_to_target  # Updated to reflect standard naming
from utils.diagnostic import get_AICc  # Consistent with diagnostics.py function name
from utils.search import golden_section
from utils.kernels import Kernel  # Import the class, not a function

class LocalScorer:
    """
    Lightweight local regression model used to score candidate bandwidths (bw) and
    smoothing parameters (K) during the backfitting process in GLASS.
    Computes local predictions, residuals, and the trace of the hat matrix.
    """
    def __init__(self, coords, X, y, bw, kernel='gaussian'):
        self.coords = coords              # n x 2 coordinate matrix
        self.X = X                        # n x p design matrix
        self.y = y.reshape(-1, 1)         # n x 1 response vector
        self.bw = bw                      # bandwidth parameter
        self.kernel = kernel              # kernel type
        self.n, self.p = X.shape
        self.predy = np.zeros((self.n, 1))
        self.resid = np.zeros((self.n, 1))
        self.tr_S = 0.0                   # trace of the hat matrix

    def fit(self):
        """ Fit local models for each observation using specified kernel. """
        for i in range(self.n):
            dists = np.sqrt(np.sum((self.coords - self.coords[i]) ** 2, axis=1))
            weights = Kernel(i, self.coords, bw=self.bw, fixed=True, function=self.kernel).kernel
            W = np.diag(weights)
            XtW = self.X.T @ W
            XtWX = XtW @ self.X
            XtWy = XtW @ self.y
            try:
                beta = np.linalg.solve(XtWX, XtWy)
                y_hat = self.X[i, :] @ beta
                self.predy[i] = y_hat
                self.resid[i] = self.y[i] - y_hat
                S_i = self.X[i, :] @ np.linalg.inv(XtWX) @ XtW[:, i]
                self.tr_S += S_i
            except np.linalg.LinAlgError:
                self.predy[i] = 0
                self.resid[i] = self.y[i]
                self.tr_S += 0
        return self

class GLASS:
    """
    Generalized Local Additive Spatial Smoothing (GLASS) model.
    Extends MGWR to support change-of-support via spatial smoothing (parameter K),
    and allows multiscale, multivariate local modeling with internal bandwidth optimization.
    """
    def __init__(self, coords, y, support_map, K_grid, bw_range=None, tol_bw=1e-3):
        self.coords = coords
        self.y = y.reshape(-1, 1)
        self.support_map = support_map
        self.K_grid = K_grid
        self.bw_range = bw_range if bw_range else (10, 150)
        self.tol_bw = tol_bw
        self.n, self.p = y.shape[0], len(support_map)
        self.K = [None] * self.p
        self.bw = [None] * self.p
        self.params = np.zeros((self.n, self.p))  # Final estimated beta surfaces
        self.X_raw = [support_map[j]['data'] for j in range(self.p)]

    def fit(self, max_iter=10, tol=1e-3):
        """
        Main model-fitting routine using iterative backfitting to jointly optimize
        bandwidths and change-of-support smoothing parameters (if needed).
        """
        last_K = [None] * self.p
        last_bw = [None] * self.p

        for iteration in range(max_iter):
            for j in range(self.p):
                best_score = np.inf
                best_K = None
                best_bw = None
                K_candidates = self.K_grid if self.support_map[j]['role'] == 'source' else [None]

                for K in K_candidates:
                    if self.support_map[j]['role'] == 'source' and 'source_coords' not in self.support_map[j]:
                        raise ValueError(f"Source coordinates must be provided for variable {j} marked as 'source'.")

                    source_coords = self.support_map[j].get('source_coords')
                    xj = smooth_source_to_target(self.X_raw[j], source_coords, self.coords, K) if K is not None else self.X_raw[j]
                    X_others = self._build_X_exclude(j)

                    def score_func(bw):
                        Xj_full = np.column_stack([X_others, xj.reshape(-1, 1)])
                        model = LocalScorer(self.coords, Xj_full, self.y, bw)
                        return get_AICc(model.fit())

                    bw_opt = golden_section(score_func, self.bw_range[0], self.bw_range[1], self.tol_bw)
                    score = score_func(bw_opt)

                    if score < best_score:
                        best_score = score
                        best_K = K
                        best_bw = bw_opt

                self.K[j] = best_K
                self.bw[j] = best_bw

            if self._has_converged(last_K, last_bw, tol):
                break
            last_K, last_bw = self.K[:], self.bw[:]

        # Final model estimation using variable-specific bandwidths
        X_final = []
        for j in range(self.p):
            if self.support_map[j]['role'] == 'source' and 'source_coords' not in self.support_map[j]:
                raise ValueError(f"Source coordinates must be provided for variable {j} marked as 'source'.")

            source_coords = self.support_map[j].get('source_coords')
            xj = smooth_source_to_target(self.X_raw[j], source_coords, self.coords, self.K[j]) if self.K[j] is not None else self.X_raw[j]
            X_final.append(xj.reshape(-1, 1))
        X = np.hstack(X_final)

        for i in range(self.n):
            dists = np.sqrt(np.sum((self.coords - self.coords[i]) ** 2, axis=1))
            for j in range(self.p):
                xj = X[:, j].reshape(-1, 1)
                weights = Kernel(i, self.coords, bw=self.bw[j], fixed=True, function='gaussian').kernel.reshape(-1, 1)
                xw = xj * weights
                XtWX = xw.T @ xj
                XtWy = xw.T @ self.y
                try:
                    beta_j = np.linalg.solve(XtWX, XtWy)
                    self.params[i, j] = beta_j.item()
                except np.linalg.LinAlgError:
                    self.params[i, j] = np.nan

    def _build_X_exclude(self, j):
        cols = []
        for k in range(self.p):
            if k != j:
                if self.support_map[k]['role'] == 'source':
                    if 'source_coords' not in self.support_map[k]:
                        raise ValueError(f"Source coordinates must be provided for variable {k} marked as 'source'.")
                    source_coords = self.support_map[k]['source_coords']
                    xk = smooth_source_to_target(
                        self.X_raw[k],
                        source_coords,
                        self.coords,
                        self.K[k]
                    )
                else:
                    xk = self.X_raw[k]
                cols.append(xk.reshape(-1, 1))
        return np.hstack(cols)

    def _has_converged(self, last_K, last_bw, tol):
        for k_new, k_old, bw_new, bw_old in zip(self.K, last_K, self.bw, last_bw):
            if k_new != k_old:
                return False
            if bw_old is None or abs(bw_new - bw_old) > tol:
                return False
        return True