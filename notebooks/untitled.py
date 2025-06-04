import numpy as np
from libpysal.weights import lat2W, Kernel
from numpy.linalg import inv
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from spglm.iwls import _compute_betas_gwr, iwls
from spglm.family import Gaussian
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor
from statsmodels.api import OLS
from mgwr.gwr import GWR, MGWR
from mgwr.sel_bw import Sel_BW
from mgwr.diagnostics import get_AICc
import libpysal as ps  # Ensure PySAL is installed
from libpysal.weights import DistanceBand, lat2W, Kernel
import geopandas as gpd
from shapely.geometry import Point, Polygon
import pandas as pd
from scipy.stats import pearsonr

from numpy.linalg import inv

import warnings
warnings.filterwarnings('ignore')



## Define autocorrelation function. 

def draw_map(N, lamb, W):
    """
    N    = number of areal units
    lamb = spatial autocorrelation parameter
    W    = spatial weights matrix
    """

    W.transform = "r"
    e = np.random.random((N, 1))
    u = inv(np.eye(N) - lamb * W.full()[0])
    u = np.dot(u, e)
    u = (u - u.mean()) / np.std(u)
    return u

def uniform(Nlat, value=3):
 
    return np.full((Nlat**2, 1), value)

uniform_beta = uniform(48)

def radial(Nlat, center_value=5, edge_value=1):
    grid = np.zeros((Nlat, Nlat))
    cx, cy = Nlat / 2, Nlat / 2
    max_dist = np.sqrt((cx)**2 + (cy)**2)

    for i in range(Nlat):
        for j in range(Nlat):
            dist = np.sqrt((i - cx)**2 + (j - cy)**2)
            weight = 1 - (dist / max_dist)  # normalized to [0, 1]
            grid[i, j] = edge_value + (center_value - edge_value) * weight

    return grid.flatten().reshape(-1, 1)

radial_beta = radial(48)

def gradient(Nlat, min_val=0, max_val=5):

    # Create row (Y) and column (X) indices
    rows, cols = np.meshgrid(np.arange(Nlat), np.arange(Nlat), indexing='ij')
    
    # Normalize and invert diagonal direction
    diag_gradient = (rows + (Nlat - 1 - cols)) / (2 * (Nlat - 1))  # NE to SW
    
    # Rescale to desired range
    beta_vals = min_val + (max_val - min_val) * diag_gradient
    return beta_vals.reshape(-1, 1)

gradient_beta = gradient(48)

## Generate beta
Nlat = 48
N = Nlat**2

W = lat2W(Nlat, Nlat, rook=True)

np.random.seed(2025)
beta_surface = draw_map(N, 0.98, W)

beta_surface_scaled = 2 * (beta_surface - beta_surface.min()) / (beta_surface.max() - beta_surface.min()) - 1

tracts_gdf48['uniform_beta'] = uniform_beta
tracts_gdf48['radial_beta'] = radial_beta
tracts_gdf48['gradient_beta'] = gradient_beta
tracts_gdf48['autocorr_beta'] = beta_surface_scaled

# Define function to create tract polygons over the area
def simulate_clustered_X(rho, num_points, background_points, num_clusters, grid_gdf, cluster_std=50, random_state=None):
   
    if random_state is not None:
        np.random.seed(random_state)

    x_range = (grid_gdf.total_bounds[0], grid_gdf.total_bounds[2])
    y_range = (grid_gdf.total_bounds[1], grid_gdf.total_bounds[3])

    centers = np.random.uniform(low=x_range[0], high=x_range[1], size=(num_clusters, 2))
    clustered_locations, _ = make_blobs(n_samples=num_points, centers=centers, cluster_std=cluster_std)

    # Clip points to stay within the study area
    clustered_locations[:, 0] = np.clip(clustered_locations[:, 0], x_range[0], x_range[1])
    clustered_locations[:, 1] = np.clip(clustered_locations[:, 1], y_range[0], y_range[1])

    background_locations = np.random.uniform(low=[x_range[0], y_range[0]],
                                             high=[x_range[1], y_range[1]],
                                             size=(background_points, 2))

    # Step 4: Combine All Points
    all_locations = np.vstack((clustered_locations, background_locations))

    ## Generate data
    Nlat = 48
    N = Nlat**2
    D = 2
    M = 3
    W = lat2W(Nlat, Nlat, rook=True)

    rawX1 = draw_map(N, rho, W).flatten()

    # newX = (rawX1-rawX1.min()) / (rawX1.max()-rawX1.min()) #* (7000 - 250) + 250 
    
    x = all_locations[:, 0]
    y = all_locations[:, 1]
    S1 = []
    cell_size = 600 / Nlat  # 12.5 in this case
    S1 = []
    for i in range(len(all_locations)):
        u = min(int(np.floor(x[i] / cell_size)), Nlat - 1)
        v = min(int(np.floor(y[i] / cell_size)), Nlat - 1)
        S1.append(rawX1[u * Nlat + v])

    return np.array(S1), np.array(all_locations)
                       
# Run Simulation with Ensured Coverage
S1, loc1  = simulate_clustered_X(rho=0.70, num_points=2000, background_points=1600, num_clusters=15, grid_gdf=tracts_gdf48,
                                 cluster_std=35, random_state=27)
S2, loc2  = simulate_clustered_X(rho=0.80, num_points=2200, background_points=1700, num_clusters=25, grid_gdf=tracts_gdf48,
                                 cluster_std=25, random_state=127)
S3, loc3  = simulate_clustered_X(rho=0.90, num_points=1800, background_points=1000, num_clusters=30, grid_gdf=tracts_gdf48,
                                 cluster_std=25, random_state=227)

# Define grid parameters and create tracts
xmin, xmax = 0, 600
ymin, ymax = 0, 600

def create_tracts(xmin, xmax, ymin, ymax, x_step, y_step, random_state=None):
    """
    Creates tract polygons and assigns a census variable (e.g., median income) to each tract.

    Parameters:
    - xmin, xmax, ymin, ymax: Bounds of the area to create tracts.
    - x_step, y_step: Dimensions of each tract (width and height).
    - random_state: Seed for reproducibility.
    """
    if random_state is not None:
        np.random.seed(random_state)
    polys = []
    ids = []
    variable_values = []
    id_counter = 0
    for x0 in np.arange(xmin, xmax, x_step):
        for y0 in np.arange(ymin, ymax, y_step):
            x1 = x0 + x_step
            y1 = y0 + y_step
            poly = Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
            polys.append(poly)
            ids.append(id_counter)
            # generate random value to represent a census variable
            value = np.random.uniform(1, 10)
            variable_values.append(round(value))
            id_counter += 1
    gdf = gpd.GeoDataFrame({'tract_id': ids, 'age': variable_values, 'geometry': polys})
    return gdf

tracts_var = create_tracts(xmin, xmax, ymin, ymax, x_step, y_step, random_state=26)

S_target = tracts_var['age'].values
loc_target = np.array([[point.x, point.y] for point in tracts_var.geometry.centroid])

# for true y relationship 
# Smooth S1 to target support
nbrs_S1 = NearestNeighbors(n_neighbors=25).fit(loc1)
smoothed_S1_to_target = np.zeros(len(S_target))

for i in range(len(S_target)):
    distances, indices = nbrs_S1.kneighbors([loc_target[i]])
    smoothed_S1_to_target[i] = np.mean(S1[indices[0]])

# Smooth S2 to target support
nbrs_S2 = NearestNeighbors(n_neighbors=25).fit(loc2)
smoothed_S2_to_target = np.zeros(len(S_target))

for i in range(len(S_target)):
    distances, indices = nbrs_S2.kneighbors([loc_target[i]])
    smoothed_S2_to_target[i] = np.mean(S2[indices[0]])

error_term = np.random.normal(0, 1.75, len(loc_target))

y = (radial_beta.flatten() * tracts_var['age'].values) + (uniform_beta.flatten() * smoothed_S1_to_target) + (beta_surface_scaled.flatten() * smoothed_S2_to_target) + error_term

df_S1 = gpd.GeoDataFrame({'S1': S1}, geometry=gpd.points_from_xy(loc1[:, 0], loc1[:, 1]))
df_S2 = gpd.GeoDataFrame({'S2': S2}, geometry=gpd.points_from_xy(loc2[:, 0], loc2[:, 1]))


# Spatial join: assign S1 points to 48x48 tracts
S1_agg = gpd.sjoin(df_S1, tracts_var, how="left", predicate="within")
S2_agg = gpd.sjoin(df_S2, tracts_var, how="left", predicate="within")

S1_agg = S1_agg.groupby("tract_id")["S1"].mean().reset_index()
S2_agg = S2_agg.groupby("tract_id")["S2"].mean().reset_index()

mgwr_df = tracts_var.merge(S1_agg, on='tract_id', how='left')
mgwr_df = mgwr_df.merge(S2_agg, on='tract_id', how='left')

mgwr_df.fillna({'S1': 0, 'S2':0}, inplace=True)

g_X  = mgwr_df[['age', 'S1', 'S2']].values
g_y = y.reshape(-1, 1)
coords = np.array([[point.x, point.y] for point in mgwr_df['geometry'].centroid])

mgwr_selector = Sel_BW(coords, g_y, g_X, multi=True, constant=False)
mgwr_bw = mgwr_selector.search(multi_bw_min=[2])
mgwr_results = MGWR(coords, g_y, g_X, mgwr_selector, constant=False).fit()


mgwr_results.summary()

tracts_gdf48['beta_S1'] = mgwr_results.params[:,0]
tracts_gdf48['beta_S2'] = mgwr_results.params[:,1]
tracts_gdf48['beta_S3'] = mgwr_results.params[:,2]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
cmap = plt.cm.Reds
cmap2 = plt.cm.RdBu_r

# Plot beta_S1
tracts_gdf48.plot(column="beta_S1", cmap=cmap, linewidth=0.1, edgecolor='k',
                  ax=axes[0])
axes[0].set_title("β₁ (S1)", fontsize=14)
axes[0].axis('off')
sm1 = cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(
    vmin=tracts_gdf48["beta_S1"].min(), vmax=tracts_gdf48["beta_S1"].max()))
sm1._A = []
fig.colorbar(sm1, ax=axes[0], orientation="vertical", fraction=0.046, pad=0.01)

# Plot beta_S2
tracts_gdf48.plot(column="beta_S2", cmap=cmap, linewidth=0.1, edgecolor='k',
                  ax=axes[1])
axes[1].set_title("β₂ (S2)", fontsize=14)
axes[1].axis('off')
sm2 = cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(
    vmin=tracts_gdf48["beta_S2"].min(), vmax=tracts_gdf48["beta_S2"].max()))
sm2._A = []
fig.colorbar(sm2, ax=axes[1], orientation="vertical", fraction=0.046, pad=0.01)

# Plot beta_S3
tracts_gdf48.plot(column="beta_S3", cmap=cmap2, linewidth=0.1, edgecolor='k',
                  ax=axes[2])
axes[2].set_title("β₃ (S3)", fontsize=14)
axes[2].axis('off')
sm3 = cm.ScalarMappable(cmap=cmap2, norm=mcolors.Normalize(
    vmin=tracts_gdf48["beta_S3"].min(), vmax=tracts_gdf48["beta_S3"].max()))
sm3._A = []
fig.colorbar(sm3, ax=axes[2], orientation="vertical", fraction=0.046, pad=0.01)

plt.tight_layout()
plt.show()






































































