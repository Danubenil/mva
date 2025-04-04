#
#
#      0===========================================================0
#      |    TP1 Basic structures and operations on point clouds    |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Second script of the practical session. Subsampling of a point cloud
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
#   Here you can define useful functions to be used in the main
#


def cloud_decimation(points, colors, labels, factor):

    indices_kept = np.arange(0, len(points), factor)
    decimated_points = points[indices_kept]
    decimated_colors = colors[indices_kept]
    decimated_labels = labels[indices_kept]
    return decimated_points, decimated_colors, decimated_labels



def grid_decimation(points, colors, labels, dl):
    # dl = voxel size
    indices_voxels = np.floor(points / dl).astype(int) # maps every points to its voxel

    voxels = {}
    voxels_color = {}
    voxels_labels = {}
    for i, ind_voxel in enumerate(indices_voxels):
        key = tuple(ind_voxel)
        if key not in voxels:
            voxels[key] = []
            voxels_color[key] = []
            voxels_labels[key] = []
        voxels[key].append(points[i])
        voxels_color[key].append(colors[i])
        voxels_labels[key].append(labels[i])
    decimated_points = []
    decimated_colors = []
    decimates_labels = []
    for key in voxels.keys():
        decimated_points.append(np.mean(np.array(voxels[key]), axis = 0))
        decimated_colors.append(np.mean(np.array(voxels_color[key]), axis = 0))
        decimates_labels.append(np.mean(np.array(voxels_labels[key]), axis = 0))
    return np.array(decimated_points), np.array(decimated_colors), np.array(decimates_labels)
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
    colors = np.vstack((data['red'], data['green'], data['blue'])).T
    labels = data['label']    

    # Decimate the point cloud
    # ************************
    #

    # Define the decimation factor
    factor = 300

    # Decimate
    t0 = time.time()
    #decimated_points, decimated_colors, decimated_labels = cloud_decimation(points, colors, labels, factor)
    decimated_points, decimated_colors, decimated_labels = grid_decimation(points, colors, labels, 0.2)

    t1 = time.time()
    print('decimation done in {:.3f} seconds'.format(t1 - t0))

    # Save
    write_ply('../decimated.ply', [decimated_points, decimated_colors, decimated_labels], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])

    
    print('Done')
