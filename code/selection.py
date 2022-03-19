#
#
#      0=============================0
#      |    TP4 Point Descriptors    |
#      0=============================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Script of the practical session
#
# ------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 13/12/2017
#


# ------------------------------------------------------------------------------------------
#
#          Imports and global variables
#      \**********************************/
#


# Import numpy package and name it "np"
import numpy as np

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from ply import write_ply, read_ply

# Import time package
import time

from tqdm import tqdm


# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#


def PCA(points):
    centroid = np.mean(points, axis=0, keepdims=True)

    cov_matrix = 1/len(points) * (centroid - points).T@(centroid - points)

    return np.linalg.eigh(cov_matrix)


def compute_local_PCA(query_points, cloud_points, radius, tree=None):

    # This function needs to compute PCA on the neighborhoods of all query_points in cloud_points

    all_eigenvalues = np.zeros((query_points.shape[0], 3))
    all_eigenvectors = np.zeros((query_points.shape[0], 3, 3))

    if tree is None:
        tree = KDTree(cloud_points)

    nearest_neighbors = tree.query_radius(query_points, radius)
    # alternatively
    # _, nearest_neighbors = tree.query(query_points, k=30)
    for i, neighbors in enumerate(nearest_neighbors):
        eigenvalues, eigenvectors = PCA(cloud_points[neighbors])
        all_eigenvalues[i] = eigenvalues
        all_eigenvectors[i] = eigenvectors
    return all_eigenvalues, all_eigenvectors


def compute_features(query_points, cloud_points, radius, tree=None):

    epsilon = 1e-6

    all_eigenvalues, all_eigenvectors = compute_local_PCA(
        query_points, cloud_points, radius, tree)
    lambda3, lambda2, lambda1 = all_eigenvalues[:, 0], all_eigenvalues[:, 1], all_eigenvalues[:, 2]
    lambda1 += epsilon  # only denominator
    normals = all_eigenvectors[:, :, 0]

    sigma1, sigma2, sigma3 = np.sqrt(lambda1), np.sqrt(lambda2), np.sqrt(lambda3)

    a1d = 1 - sigma2/sigma1
    a2d = (sigma2 - sigma3)/sigma1
    a3d = sigma3/sigma1

    V = sigma1*sigma2*sigma3

    return normals, a1d, a2d, a3d, V


def dimensionality_features(query_points, cloud_points, radius, tree=None):
    normals, a1d, a2d, a3d, V = compute_features(query_points, cloud_points, radius, tree)
    d_star = np.argmax([a1d, a2d, a3d], axis=0)
    d_star += 1  # python indexing
    return normals, d_star, V


def entropy(query_points, cloud_points, radius, tree=None):
    _, a1d, a2d, a3d, _ = compute_features(query_points, cloud_points, radius, tree)
    return - a1d*np.log(a1d) - a2d*np.log(a2d) - a3d*np.log(a3d)


def optimal_neighborhood_radius(query_points, cloud_points, radius_mini, radius_maxi, tree=None):

    # first, create the radii that will be tested
    iteration = np.linspace(0, 10, 16)  # best trade-off
    iteration **= 2 # we square the gaps
    iteration = radius_mini + \
        (radius_maxi - radius_mini)*iteration/max(iteration)

    unpredictability = np.zeros((16, query_points.shape[0]))

    if tree is None:
        tree = KDTree(cloud_points)

    for i, radius in enumerate(tqdm(iteration)):
        unpredictability[i] = entropy(query_points, cloud_points, radius, tree)

    return radius_mini + (radius_maxi-radius_mini)*np.argmin(unpredictability, axis=0)/16

# ------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
#
#   Here you can define the instructions that are called when you execute this file
#
if __name__ == '__main__':

    # Normal computation
    # ******************
    if False:

        # Load cloud as a [N x 3] matrix
        cloud_path = '../data/Lille_street_small.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

        # Compute PCA on the whole cloud
        all_eigenvalues, all_eigenvectors = compute_local_PCA(
            cloud, cloud, 0.50)
        normals = all_eigenvectors[:, :, 0]

        # Save cloud with normals
        write_ply('../Lille_street_small_normals.ply',
                  (cloud, normals), ['x', 'y', 'z', 'nx', 'ny', 'nz'])
    if False:

        # Load cloud as a [N x 3] matrix
        cloud_path = '../data/Notre_Dame_Des_Champs_1.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

        # Compute PCA on the whole cloud
        query = cloud[:1000]
        normals, a1d, a2d, a3d, V = compute_features(query, cloud, 0.5)
        print(len(a1d))

        # Save cloud with normals
        # write_ply('../Lille_street_small.ply',
        # (cloud, normals), ['x', 'y', 'z', 'nx', 'ny', 'nz'])
        # write_ply('../Lille_small_feat.ply', [query, vert, lin, plan, spher], [
        # 'x', 'y', 'z', 'vert', 'lin', 'plan', 'spher'])

    if True:
        # test functions

        # Load cloud as a [N x 3] matrix
        cloud_path = '../data/bunny_original.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T
        query = cloud[:1000]
        # normals, d_star, V = dimensionality_features(query, cloud, 0.5)
        # print(entropy(cloud, cloud, 0.005))
        print(np.shape(query))

        radii = optimal_neighborhood_radius(query, cloud, 0.1, 1)

        print(radii)
        print(np.shape(radii))
