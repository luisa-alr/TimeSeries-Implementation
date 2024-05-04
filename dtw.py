import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt


# Function to compute the cost matrix
def compute_cost_matrix(X, Y):
    X, Y = np.atleast_2d(X, Y)  # Ensure X and Y are at least 2-dimensional
    cost_matrix = scipy.spatial.distance.cdist(
        X.T, Y.T
    )  # Compute the pairwise distances between columns of X and Y
    return cost_matrix


# Function to compute the accumulated cost matrix
def compute_distance(cost_matrix):
    N = cost_matrix.shape[0]
    M = cost_matrix.shape[1]
    dist = np.zeros((N, M))  # Initialize the accumulated cost matrix
    dist[0, 0] = cost_matrix[
        0, 0
    ]  # Initialize the first cell of the accumulated cost matrix
    # Fill the first row and column of the accumulated cost matrix
    for n in range(1, N):
        dist[n, 0] = dist[n - 1, 0] + cost_matrix[n, 0]
    for m in range(1, M):
        dist[0, m] = dist[0, m - 1] + cost_matrix[0, m]
    # Fill the rest of the accumulated cost matrix using dynamic programming
    for n in range(1, N):
        for m in range(1, M):
            dist[n, m] = cost_matrix[n, m] + min(
                dist[n - 1, m], dist[n, m - 1], dist[n - 1, m - 1]
            )
    return dist


# Function to compute the optimal warping path
def compute_path(dist):
    N = dist.shape[0]
    M = dist.shape[1]
    n = N - 1
    m = M - 1
    path = [(n, m)]  # Start from the bottom-right corner of the accumulated cost matrix
    # Backtrack through the accumulated cost matrix to find the optimal warping path
    while n > 0 or m > 0:
        if n == 0:
            cell = (0, m - 1)
        elif m == 0:
            cell = (n - 1, 0)
        else:
            val = min(dist[n - 1, m - 1], dist[n - 1, m], dist[n, m - 1])
            if val == dist[n - 1, m - 1]:
                cell = (n - 1, m - 1)
            elif val == dist[n - 1, m]:
                cell = (n - 1, m)
            else:
                cell = (n, m - 1)
        path.append(cell)
        (n, m) = cell
    path.reverse()  # Reverse the path to get the correct order
    return np.array(path)


# Input time series data
# OR implement code here to read time series from data frame
X = [32, 36, 27, 37, 35, 40, 34, 33, 25, 29]
Y = [31, 32, 32, 30, 37, 39, 29, 34, 25, 26]
N = len(X)
M = len(Y)

# Plot the input time series data
plt.figure(figsize=(6, 2))
plt.plot(X, c="k", label="$X$")
plt.plot(Y, c="b", label="$Y$")
plt.legend()
plt.tight_layout()
plt.show()

# Compute the cost matrix
cost_matrix = compute_cost_matrix(X, Y)
print("Cost matrix =", cost_matrix, sep="\n")

# Compute the accumulated cost matrix
dist = compute_distance(cost_matrix)
print("Accumulated cost matrix distance =", dist, sep="\n")
print("Dynamic Time Warping distance DTW(X, Y) =", dist[-1, -1])

# Compute the optimal warping path
path = compute_path(dist)
print("Optimal warping path =", path.tolist())

# Plot the cost matrix with the optimal warping path
path = np.array(path)
plt.figure(figsize=(9, 3))
plt.subplot(1, 2, 1)
plt.imshow(cost_matrix, cmap="gray_r", origin="lower", aspect="equal")
plt.plot(path[:, 1], path[:, 0], marker="o", color="r")
plt.clim([0, np.max(cost_matrix)])
plt.colorbar()
plt.title("$C$ with optimal warping path")
plt.xlabel("Sequence Y")
plt.ylabel("Sequence X")
plt.show()

# Plot the accumulated cost matrix with the optimal warping path
plt.subplot(1, 2, 2)
plt.imshow(dist, cmap="gray_r", origin="lower", aspect="equal")
plt.plot(path[:, 1], path[:, 0], marker="o", color="r")
plt.clim([0, np.max(dist)])
plt.colorbar()
plt.title("$D$ with optimal warping path")
plt.xlabel("Sequence Y")
plt.ylabel("Sequence X")
plt.show()
