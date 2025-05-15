#
#
#      0===========================================================0
#      |              TP4 Surface Reconstruction                   |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Jean-Emmanuel DESCHAUD - 15/01/2024
#


# Import numpy package and name it "np"
import numpy as np

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from ply import write_ply, read_ply

# Import time package
import time

from skimage import measure

import trimesh


# Hoppe surface reconstruction
def compute_hoppe(points,normals,scalar_field,grid_resolution,min_grid,size_voxel):
    
    kd = KDTree(points)

    for i in range((grid_resolution)):
        for j in range((grid_resolution)):
            for k in range((grid_resolution)):
                voxel = np.array([i*size_voxel + min_grid[0], j*size_voxel + min_grid[1], k*size_voxel + min_grid[2]])
                # get neighbors indices 
                indices_plus_proches = kd.query(voxel[None, :])[1]
                neighbor = points[indices_plus_proches].flatten()
                normals_neighbors = normals[indices_plus_proches]
                # compute f value
                scalar_field[i, j, k] = normals_neighbors @ (voxel - neighbor)

    return None 
    

# IMLS surface reconstruction
def compute_imls(points,normals,scalar_field,grid_resolution,min_grid,size_voxel,knn):

    kd = KDTree(points)
    h =0.01

    for i in range((grid_resolution)):
        for j in range((grid_resolution)):
            for k in range((grid_resolution)):
                voxel = np.array([i*size_voxel + min_grid[0], j*size_voxel + min_grid[1], k*size_voxel + min_grid[2]])
                V = np.full((knn, 3) ,voxel)  # create a matrix to use for matrix multiplication (faster)
                # get neighbors indices 
                indices_plus_proches = kd.query(voxel[None, :], k=knn)[1][0]
                P = points[indices_plus_proches]
                N = normals[indices_plus_proches]
                Theta = np.exp(- (np.linalg.norm((V - P), axis=1) / h) ** 2)
                scalar_field[i, j, k] = np.sum(N @ (V - P).T @ Theta) / np.sum(Theta)
                
    return None



if __name__ == '__main__':

    t0 = time.time()
    
    # Path of the file
    file_path = '../data/bunny_normals.ply'

    # Load point cloud
    data = read_ply(file_path)

    # Concatenate data
    points = np.vstack((data['x'], data['y'], data['z'])).T
    normals = np.vstack((data['nx'], data['ny'], data['nz'])).T

	# Compute the min and max of the data points
    min_grid = np.amin(points, axis=0)
    max_grid = np.amax(points, axis=0)

	# Increase the bounding box of data points by decreasing min_grid and inscreasing max_grid
    min_grid = min_grid - 0.10*(max_grid-min_grid)
    max_grid = max_grid + 0.10*(max_grid-min_grid)

	# grid_resolution is the number of voxels in the grid in x, y, z axis
    grid_resolution = 128 #16
    size_voxel = max([(max_grid[0]-min_grid[0])/(grid_resolution-1),(max_grid[1]-min_grid[1])/(grid_resolution-1),(max_grid[2]-min_grid[2])/(grid_resolution-1)])
    print("size_voxel: ", size_voxel)
	
	# Create a volume grid to compute the scalar field for surface reconstruction
    scalar_field = np.zeros((grid_resolution,grid_resolution,grid_resolution),dtype = np.float32)

	# Compute the scalar field in the grid
    #compute_hoppe(points,normals,scalar_field,grid_resolution,min_grid,size_voxel)
    compute_imls(points,normals,scalar_field,grid_resolution,min_grid,size_voxel,30)

	# Compute the mesh from the scalar field based on marching cubes algorithm
    verts, faces, normals_tri, values_tri = measure.marching_cubes(scalar_field, level=0.0, spacing=(size_voxel,size_voxel,size_voxel))
    verts += min_grid

    num_triangles = faces.shape[0]
    print(f"Number of triangles: {num_triangles}")

    # Export the mesh in ply using trimesh lib
    mesh = trimesh.Trimesh(vertices = verts, faces = faces)
    mesh.export(file_obj='../bunny_mesh_imls_128.ply', file_type='ply')
	
    print("Total time for surface reconstruction : ", time.time()-t0)
	# for imls 128 it took 373.95 seconds            
    # for imls 16 it took 0.88 seconds       
    # for hoppe 128 it took 229.30 seconds              
    # for hoppe 16 it took 0.68 seconds       



