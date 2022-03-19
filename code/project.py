"""Compute a better version of the ICP"""

import numpy as np

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

from ply import read_ply, write_ply

from selection import dimensionality_features, optimal_neighborhood_radius, entropy



def selection(query, point_cloud, method, tree=None, optimal_radii=None):
    """Selecting the points we will use for the ICP"""
    query_points = []
    radius = []
    match method:
        case "all":
            query_points = point_cloud
        case "random": # we keep only 10% of the points
            nb_points = np.shape(point_cloud)[1]
            query_indexes = np.random.randint(nb_points, size=int(0.1*nb_points))
            query_points = point_cloud.T[query_indexes]
        case "high_entropy_06": # select all points with an entropy higher than 0.6
            if optimal_radii is None:
                optimal_radii = optimal_neighborhood_radius(query.T, point_cloud.T, min_radii, max_radii, tree)
            entropies = entropy(query.T, point_cloud.T, optimal_radii, tree)
            for i, entropy_value in enumerate(entropies):
                if entropy_value > 0.6:
                    query_points.append(query.T[i])
                    radius.append(optimal_radii[i])
        case "high_entropy_07": # select all points with an entropy higher than 0.7
            if optimal_radii is None:
                optimal_radii = optimal_neighborhood_radius(query.T, point_cloud.T, min_radii, max_radii, tree)
            entropies = entropy(query.T, point_cloud.T, optimal_radii, tree)
            for i, entropy_value in enumerate(entropies):
                if entropy_value > 0.7:
                    query_points.append(query.T[i])
                    radius.append(optimal_radii[i])
        case "d*": # select points with non-linear nor scattered behavior
            if optimal_radii is None:
                optimal_radii = optimal_neighborhood_radius(query.T, point_cloud.T, min_radii, max_radii, tree)
            _, d_star, _ = dimensionality_features(query.T, point_cloud.T, optimal_radii, tree)
            for i, d in enumerate(d_star):
                if d == 2:
                    query_points.append(query.T[i])
                    radius.append(optimal_radii[i])
    query_points = np.array(query_points)
    radius = np.array(radius)
    return query_points.T, radius, optimal_radii 


def weighting_normal(normal1, normal2):
    return np.dot(normal1, normal2)

def weighting_ominvariance(V_1, V_2, d_max):
    return 1 - np.abs(V_1 - V_2)/d_max

def weighting_default(query_points, ref_neighbors, d_max):
    return  1 - np.sqrt(np.abs(np.einsum('ij, ij->i', query_points, ref_neighbors)))/d_max
    # return  1 - np.sqrt(np.abs([sum(i*j for i, j in zip(a, b)) for a, b in zip(query_points.T, ref_neighbors)]))/d_max

def weighting(query_points, radius, ref_neighbors, point_cloud, weighting_method):
    match weighting_method:
        case "default":
            d_max = np.sqrt(max(np.abs([np.dot(query_points.T[i], ref_neighbors[i])
                                        for i in range(len(query_points.T))])))
            distances = weighting_default(query_points.T, ref_neighbors, d_max)
        case "normal":
            query_normals, _, _ = dimensionality_features(query_points.T, point_cloud.T, radius)
            ref_normals, _, _ = dimensionality_features(ref_neighbors.T, point_cloud.T, radius)
            distances = weighting_normal(query_normals, ref_normals)
        case "omnivariance":
            _, _, V_query = dimensionality_features(query_points.T, point_cloud.T, radius)
            _, _, V_ref_neighbors = dimensionality_features(ref_neighbors, point_cloud.T, radius)
            d_max = max([np.abs(V_query[i] - V_ref_neighbors[i]) for i in range(len(query_points))])
            distances = weighting_ominvariance(V_query, V_ref_neighbors, d_max)
    return distances



def rejection_rank(distance, percentage):
    indexes_to_keep = []
    threshold = np.quantile(distance, percentage)
    for i, distance_value in enumerate(distance):
        if distance_value >= threshold:
            indexes_to_keep.append(i)
    return indexes_to_keep

def rejection_distance(distance):
    indexes_to_keep = []
    max_distance_allowed = 2.5*np.std(distance)
    for i, distance_value in enumerate(distance):
        if distance_value >= max_distance_allowed:
            indexes_to_keep.append(i)
    return indexes_to_keep

def rejection(distances, rejection_method):
    match rejection_method:
        case "rank":
            points_to_keep = rejection_rank(distances, 0.7)
        case "distance":
            points_to_keep = rejection_distance(distances)
    return points_to_keep


def center_cloud(point_cloud):
    p = np.mean(point_cloud, axis=1, keepdims=True)
    centered_point_cloud = point_cloud - p
    return p, centered_point_cloud

def best_rigid_transform(data, ref):
    '''
    Computes the least-squares best-fit transform that maps corresponding points data to ref.
    Inputs :
        data = (d x N) matrix where "N" is the number of points and "d" the dimension
         ref = (d x N) matrix where "N" is the number of points and "d" the dimension
    Returns :
           R = (d x d) rotation matrix
           T = (d x 1) translation vector
           Such that R * data + T is aligned on ref
    '''

    R = np.eye(data.shape[0])
    T = np.zeros((data.shape[0], 1))

    p_data, centered_data = center_cloud(data)

    p_ref, centered_ref = center_cloud(ref)

    H = centered_data@centered_ref.T

    u, _, vh = np.linalg.svd(H)

    v = vh.T

    R = v@u.T

    if np.linalg.det(R) < 0:
        u[:, -1] *= -1

        R = v@u.T

    T = p_ref - R@p_data

    return R, T



def turbo_ICP(data, ref, max_iter, RMS_threshold, selection_method, weighting_method, rejection_method):
    data_aligned = np.copy(data)
    optimal_radii = None # to not have to recompute it every time
    # selection step
    tree = KDTree(ref.T)
    iter_count = 0
    RMS_list = [2*RMS_threshold]
    while iter_count < max_iter and RMS_list[-1] > RMS_threshold:
        iter_count += 1

        data_aligned_tree = KDTree(data_aligned.T)

        query_points, radius, optimal_radii = selection(data_aligned, data_aligned, selection_method, data_aligned_tree, optimal_radii)
        _, nearest_neighbors = tree.query(query_points.T, 1)
        ref_neighbors = np.array([liste[0] for liste in ref.T[nearest_neighbors]])

        _, all_neighbors = tree.query(data_aligned.T, 1)
        all_neighbors = np.array([liste[0] for liste in ref.T[all_neighbors]])

        # weighting step
        distances = weighting(query_points, radius, ref_neighbors, data_aligned, weighting_method)

        pts_before_rejection = len(query_points.T)
        # rejecting_step
        points_to_keep = rejection(distances, rejection_method)

        print("points kept : ", round(len(points_to_keep)/pts_before_rejection*100), "%")

        R, T = best_rigid_transform(query_points.T[points_to_keep].T, ref_neighbors[points_to_keep].T)
        data_aligned = R@data_aligned + T
        distances2 = np.sum(np.power(data_aligned - all_neighbors.T, 2), axis=0)
        RMS = np.sqrt(np.mean(distances2))
        print(RMS)
        RMS_list.append(RMS)

    return data_aligned, RMS_list[1:]


if __name__ == "__main__":
    min_radii = 0.01
    max_radii = 0.5
    if True:
        # Cloud paths
        bunny_o_path = '../data/bunny_original.ply'
        bunny_r_path = '../data/bunny_perturbed.ply'

        # Load clouds
        bunny_o_ply = read_ply(bunny_o_path)
        bunny_r_ply = read_ply(bunny_r_path)
        bunny_o = np.vstack((bunny_o_ply['x'], bunny_o_ply['y'], bunny_o_ply['z']))
        bunny_r = np.vstack((bunny_r_ply['x'], bunny_r_ply['y'], bunny_r_ply['z']))


        data_aligned, RMS = turbo_ICP(bunny_r, bunny_o, 100, 1e-4, "d*", "default", "distance")
        print(RMS)

        # Save cloud
        write_ply('../data/bunny_r_opt', [data_aligned.T], ['x', 'y', 'z'])


    # If statement to skip this part if wanted
    if False:

        # Cloud paths
        bunny_o_path = '../data/Notre_Dame_Des_Champs_1.ply'
        bunny_p_path = '../data/Notre_Dame_Des_Champs_2.ply'

        # Load clouds
        bunny_o_ply = read_ply(bunny_o_path)
        bunny_p_ply = read_ply(bunny_p_path)
        bunny_o = np.vstack((bunny_o_ply['x'], bunny_o_ply['y'], bunny_o_ply['z']))
        bunny_p = np.vstack((bunny_p_ply['x'], bunny_p_ply['y'], bunny_p_ply['z']))

        # Apply ICP
        # bunny_p_opt, RMS_list = turbo_ICP(bunny_p, bunny_o, 100, 1e-4, sampling_limit)
        print(np.shape(bunny_o), np.shape(bunny_p))
        data_aligned, RMS = turbo_ICP(bunny_o, bunny_p, 100, 1e-4, "d*", "default", "distance")
