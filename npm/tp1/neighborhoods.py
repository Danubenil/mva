#
#
#      0===========================================================0
#      |    TP1 Basic structures and operations on point clouds    |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Third script of the practical session. Neighborhoods in a point cloud
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
import random
import numpy as np
from tqdm import tqdm
# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from ply import write_ply, read_ply

# Import time package
import time
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define useful functions to be used in the main
#


def brute_force_spherical(queries, supports, radius):
    neighborhoods = []
    for query in queries:
        distances =  np.linalg.norm(supports - query, axis = 1)
        indices_kept = np.where((distances <= radius ) & (distances > 0))
        neighborhoods.append(supports[indices_kept] )
    

    return neighborhoods


def brute_force_KNN(queries, supports, k):

    neighborhoods = []
    for query in queries:
        distances =  np.linalg.norm(supports - query, axis = 1)
        indices_kept = np.argsort(distances)[1: k + 1]

        neighborhoods.append(supports[indices_kept] )
    

    return neighborhoods





# ------------------------------------------------------------------------------------------
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

    # Brute force neighborhoods
    # *************************
    #

    # If statement to skip this part if you want
    if True:

        # Define the search parameters
        neighbors_num = 100
        radius = 0.2
        num_queries = 10

        # Pick random queries
        random_indices = np.random.choice(points.shape[0], num_queries, replace=False)
        queries = points[random_indices, :]

        # Search spherical
        t0 = time.time()
        neighborhoods = brute_force_spherical(queries, points, radius)
        t1 = time.time()

        # Search KNN      
        neighborhoods = brute_force_KNN(queries, points, neighbors_num)
        t2 = time.time()

        # Print timing results
        print('{:d} spherical neighborhoods computed in {:.3f} seconds'.format(num_queries, t1 - t0))
        print('{:d} KNN computed in {:.3f} seconds'.format(num_queries, t2 - t1))

        # Time to compute all neighborhoods in the cloud
        total_spherical_time = points.shape[0] * (t1 - t0) / num_queries
        total_KNN_time = points.shape[0] * (t2 - t1) / num_queries
        print('Computing spherical neighborhoods on whole cloud : {:.0f} hours'.format(total_spherical_time / 3600))
        print('Computing KNN on whole cloud : {:.0f} hours'.format(total_KNN_time / 3600))

 



    # KDTree neighborhoods
    # ********************
    #



        # Define the search parameters
    
    random.seed(42)
    num_queries = 1000
    indices = np.arange(0, len(points))
    np.random.shuffle(points)
    queries = points[indices[0:num_queries]]
    model = KDTree(points)
    if False:
        times = []
        n_leaves = 40
        for i in range(1, n_leaves):
            
            tic = time.time()
            ind = model.query_radius(queries, r = radius)
            tac = time.time()
            times.append(tac - tic)
        plt.xlabel("Number of leaves")
        plt.ylabel("Duration")
        plt.title("Time complexity in function of the number of leaves")
        plt.plot(np.arange(1, n_leaves), times)
        plt.legend()
        plt.savefig("TimeComplexity")
        plt.show()

        print(f"Fastest search for leaf_size {np.argmin(times) + 1} with time complexity of {np.min(times)} seconds.")
            
    leaf_size = 15
    #radiuses = np.linspace(0,1, 100)
    #time_computation = []
    #for radius in tqdm(radiuses):
    #    tic = time.time()
    #    ind = model.query_radius(queries, r = radius)
    #    tac = time.time()
    #    time_computation.append(tac - tic )
    #plt.xlabel("Radius (m)")
    #plt.ylabel("Computation time (sec)")
    #plt.title("Computation time as a function of the radius")
    #plt.plot(radiuses, time_computation)
    #plt.legend()
    #plt.savefig("TimeComplexityRadius")
    #plt.show()
    tic = time.time()
    ind = model.query_radius(points, r = 0.2)
    tac = time.time()
    print(f"Computation time of the 20cm spherical neighborhood of all the points : {tac - tic:.2f} seconds.")


        