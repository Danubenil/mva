#
#
#      0===========================================================0
#      |                      TP6 Modelisation                     |
#      0===========================================================0
#
#
#------------------------------------------------------------------------------------------
#
#      Plane detection with RANSAC
#
#------------------------------------------------------------------------------------------
#
#      Xavier ROYNARD - 19/02/2018
#


#------------------------------------------------------------------------------------------
#
#          Imports and global variables
#      \**********************************/
#


# Import numpy package and name it "np"
from genericpath import isfile
import numpy as np
from tqdm import tqdm
import pickle
# Import functions to read and write ply files
from ply import write_ply, read_ply
import os
# Import time package
import time

from sklearn.neighbors import KDTree

#------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#

# QUESTION 2



def compute_plane(points):
    p0, p1, p2 = points[0], points[1], points[2]

    u = p1 - p0
    v = p2 - p0
    normal_plane = np.cross(u, v)
    normal_plane /= np.linalg.norm(normal_plane)
    
    return p0, normal_plane



def in_plane(points, pt_plane, normal_plane, threshold_in=0.1):

    
    vectors = points - pt_plane.T  
    dot_prod = np.abs(vectors @ normal_plane) 
    
    return dot_prod < threshold_in 



def RANSAC(points, nb_draws=100, threshold_in=0.1):

    best_vote = 3
    best_pt_plane = np.zeros((3,1))
    best_normal_plane = np.zeros((3,1))
    
    for i in range(nb_draws):

        pts = points[np.random.choice(len(points), size=3, replace = False)]
        point_plane, normal_plane = compute_plane(pts)
        indexes = in_plane(points, point_plane, normal_plane, threshold_in)
        vote = np.sum(indexes)
        
        if best_vote < vote:
            best_vote = vote
            best_pt_plane = point_plane
            best_normal_plane = normal_plane
            
    return best_pt_plane, best_normal_plane, best_vote


def recursive_RANSAC(points, nb_draws=100, threshold_in=0.1, nb_planes=2,  indices = None):
    if nb_planes == 0 or len(points) == 0:
        return np.array([], dtype = np.int32), indices, np.array([], dtype = np.int32)
    if indices is None:
        indices = np.arange(len(points))
        points_considered = points
    else:
        points_considered = points[indices]
    best_pt_plane, best_normal_plane, _ = RANSAC(points_considered, nb_draws = nb_draws, threshold_in= threshold_in)

    points_in_plane = in_plane(points, best_pt_plane, best_normal_plane, threshold_in = threshold_in)
    plane_inds = np.intersect1d(points_in_plane.nonzero()[0], indices)
    remaining_inds = np.intersect1d((1-points_in_plane).nonzero()[0], indices)
    new_plane_inds, new_remaining_inds, labels= recursive_RANSAC(points, nb_draws = nb_draws, threshold_in = threshold_in, indices = remaining_inds, nb_planes = nb_planes - 1)
    labels = np.hstack((labels, [nb_planes - 1] * len(plane_inds)))
    plane_inds = np.hstack((new_plane_inds, plane_inds))
    
    return plane_inds, new_remaining_inds, labels







#### CODE FROM TP 3 ####
def PCA(points):

    barycenter = np.mean(points, axis=0)
    N = len(points)
    P = points - barycenter.reshape(1, -1) 
    

    cov = 1/N * (P.T @ P)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    return eigenvalues, eigenvectors



def compute_local_PCA(query_points, cloud_points, radius):

    # This function needs to compute PCA on the neighborhoods of all query_points in cloud_points

    all_eigenvalues = np.zeros((query_points.shape[0], 3))
    all_eigenvectors = np.zeros((query_points.shape[0], 3, 3))

    tree = KDTree(cloud_points)
    #neighborhoods = tree.query_radius(query_points, radius)
    neighborhoods = tree.query(query_points, k=30)
    
    for i in tqdm(range(len(neighborhoods[1]))) : 
        neighborhood = neighborhoods[1][i]
        neighbors = cloud_points[neighborhood]
        all_eigenvalues[i, :], all_eigenvectors[i, :] = PCA(neighbors)
        
    return all_eigenvalues, all_eigenvectors





#### CODE FROM TP 3 ####


# Question 4








def in_plane(points, pt_plane, normal_plane, threshold_in=0.1, threshold_normal = 20):



    
    vectors = points - pt_plane.T  
    dot_prod = np.abs(vectors @ normal_plane) 
    res = dot_prod < threshold_in
    indices_candidats = np.where(res)[0]
    
    deviations = np.arccos(normals[indices_candidats] @ normal_plane)
    indices_to_remove = indices_candidats[np.where(np.rad2deg(deviations) > threshold_normal)[0]]
    res[indices_to_remove] = False
    return res




#------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
# 
#   Here you can define the instructions that are called when you execute this file
#

if __name__ == '__main__':


    # Load point cloud
    # ****************
    #
    #   Load the file '../data/indoor_scan.ply'
    #   (See read_ply function)
    #

    # Path of the file
    file_path = '../data/indoor_scan.ply'

    # Load point cloud
    data = read_ply(file_path)

    # Concatenate data
    points = np.vstack((data['x'], data['y'], data['z'])).T
    colors = np.vstack((data['red'], data['green'], data['blue'])).T
    labels = data['label']
    nb_points = len(points)
    

    global normals
    indices_points = np.arange(len(points))
    # Suppress the file if you change point clouds
    if os.path.isfile("../data/normals.data"):
        with open("../data/normals.data", "rb") as f:
            normals = pickle.load(f)
    else:
        with open("../data/normals.data", "wb") as f:

            _, all_eigenvectors = compute_local_PCA(points, points, 0.50)
            normals = all_eigenvectors[:, :, 0]
            normals = normals / np.linalg.norm(normals, axis = 1)[:, None]
            pickle.dump(normals, f)
    # Computes the plane passing through 3 randomly chosen points
    # ************************
    #
    
    print('\n--- 1) and 2) ---\n')
    
    # Define parameter
    threshold_in = 0.10

    # Take randomly three points
    pts = points[np.random.randint(0, nb_points, size=3)]
    
    # Computes the plane passing through the 3 points
    t0 = time.time()
    pt_plane, normal_plane = compute_plane(pts)
    t1 = time.time()
    print('plane computation done in {:.3f} seconds'.format(t1 - t0))
    
    # Find points in the plane and others
    t0 = time.time()
    points_in_plane = in_plane(points, pt_plane, normal_plane, threshold_in)
    t1 = time.time()
    print('plane extraction done in {:.3f} seconds'.format(t1 - t0))
    plane_inds = points_in_plane.nonzero()[0]
    remaining_inds = (1-points_in_plane).nonzero()[0]

    # Save extracted plane and remaining points
    write_ply('../plane.ply', [points[plane_inds], colors[plane_inds], labels[plane_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    write_ply('../remaining_points_plane.ply', [points[remaining_inds], colors[remaining_inds], labels[remaining_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    
    # Computes the best plane fitting the point cloud
    # ***********************************
    #
    #
    
    print('\n--- 3) ---\n')

    # Define parameters of RANSAC
    nb_draws = 100
    threshold_in = 0.10

    # Find best plane by RANSAC
    t0 = time.time()
    best_pt_plane, best_normal_plane, best_vote = RANSAC(points, nb_draws, threshold_in)
    t1 = time.time()
    print('RANSAC done in {:.3f} seconds'.format(t1 - t0))
    
    # Find points in the plane and others
    points_in_plane = in_plane(points, best_pt_plane, best_normal_plane, threshold_in)
    plane_inds = points_in_plane.nonzero()[0]
    remaining_inds = (1-points_in_plane).nonzero()[0]
    print(plane_inds.shape)
    print(remaining_inds.shape)
    # Save the best extracted plane and remaining points
    write_ply('../best_plane.ply', [points[plane_inds], colors[plane_inds], labels[plane_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    write_ply('../remaining_points_best_plane.ply', [points[remaining_inds], colors[remaining_inds], labels[remaining_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    

    # Find "all planes" in the cloud
    # ***********************************
    #
    #
    
    print('\n--- 4) ---\n')
    
    # Define parameters of recursive_RANSAC
    nb_draws = 100
    threshold_in = 0.05
    nb_planes = 10
    
    # Recursively find best plane by RANSAC
    t0 = time.time()
    plane_inds, remaining_inds, plane_labels = recursive_RANSAC(points, nb_draws, threshold_in, nb_planes)
    t1 = time.time()
    print(plane_inds.shape, " ", plane_inds.dtype, " : ", plane_inds[:10])
    print(remaining_inds.shape, " : ", remaining_inds[:10])
    print(plane_labels.shape, " : ", plane_labels[:10])
    assert np.intersect1d(plane_inds, remaining_inds).size == 0
    print('recursive RANSAC done in {:.3f} seconds'.format(t1 - t0))
                
    # Save the best planes and remaining points
    write_ply('../best_planes.ply', [points[plane_inds], colors[plane_inds], labels[plane_inds], plane_labels.astype(np.int32)], ['x', 'y', 'z', 'red', 'green', 'blue', 'label', 'plane_label'])
    write_ply('../remaining_points_best_planes.ply', [points[remaining_inds], colors[remaining_inds], labels[remaining_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    
    
    
    print('Done')
    