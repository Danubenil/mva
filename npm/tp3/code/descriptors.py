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

# Import library to plot in python
from matplotlib import pyplot as plt

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from ply import write_ply, read_ply

# Import time package
import time


# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#



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
    
    for i, neighborhood in enumerate(neighborhoods[1]) : 

        neighbors = cloud_points[neighborhood]
        all_eigenvalues[i, :], all_eigenvectors[i, :] = PCA(neighbors)
        
    return all_eigenvalues, all_eigenvectors


def compute_features(query_points, cloud_points, radius):
    
    all_eigenvalues, all_eigenvectors = compute_local_PCA(query_points, cloud_points, radius)

    e_z = [0, 0, 1]
    normals = all_eigenvectors[:, :, 0] 
    normals_scaled = normals / np.linalg.norm(normals, axis = 1)[:, None]
    term = np.abs(normals_scaled @ e_z)
    eps = 1e-8
    verticality = 2 * np.arccos(term) / np.pi
    linearity = 1 - (all_eigenvalues[:, 1] / (all_eigenvalues[:, 0] + eps))
    linearity = (linearity - np.min(linearity)) / (np.max(linearity) - np.min(linearity))
    planarity = (all_eigenvalues[:, 1] - all_eigenvalues[:, 2]) / (all_eigenvalues[:, 0] + eps)
    planarity = (planarity - np.min(planarity)) / (np.max(planarity) - np.min(planarity))
    sphericity = all_eigenvalues[:, 2] / (all_eigenvalues[:, 0] + eps)
    sphericity = (sphericity - np.min(sphericity)) / (np.max(sphericity) - np.min(sphericity))
    return verticality, linearity, planarity, sphericity


# ------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
# 
#   Here you can define the instructions that are called when you execute this file
#

if __name__ == '__main__':

    # PCA verification
    # ****************
    if False:

        # Load cloud as a [N x 3] matrix
        cloud_path = '../data/Lille_street_small.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

        # Compute PCA on the whole cloud
        eigenvalues, eigenvectors = PCA(cloud)

        # Print your result
        print(eigenvalues)

        # Expected values :
        #
        #   [lambda_3; lambda_2; lambda_1] = [ 5.25050177 21.7893201  89.58924003]
        #
        #   (the convention is always lambda_1 >= lambda_2 >= lambda_3)
        #

		
    # Normal computation
    # ******************
    if False:

        # Load cloud as a [N x 3] matrix
        cloud_path = '../data/Lille_street_small.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

        # Compute PCA on the whole cloud
        all_eigenvalues, all_eigenvectors = compute_local_PCA(cloud, cloud, 0.50)
        normals = all_eigenvectors[:, :, 0]

        # Save cloud with normals
        write_ply('../Lille_street_small_normals.ply', (cloud, normals), ['x', 'y', 'z', 'nx', 'ny', 'nz'])
		
    # BONUS QUESTION 
    if True : 

        # Load cloud as a [N x 3] matrix
        cloud_path = '../data/Lille_street_small.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

        # Compute PCA on the whole cloud
        v, l, p, s = compute_features(cloud, cloud, 0.50)
        print(v)
        write_ply("../Lille_street_small_features.ply", (cloud, v, l, p, s), ["x", "y", "z", "v", "l", "p", "s"] )