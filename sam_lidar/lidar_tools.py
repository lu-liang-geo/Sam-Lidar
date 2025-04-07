import os
import laspy
import rasterio
import numpy as np
import pandas as pd
import supervision as sv
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances


def rasterize_lidar(lidar_folder, rgb_folder, filename, boxes=None, 
                    classifications={'Ground':0,'High Vegetation':1}, img_size=(400,400)):
    '''
    Indexes LiDAR points by the pixel they fall into in the corresponding RGB raster image. LiDAR
    points classified as "High Vegetation" are then further labeled according to which tree(s)
    they are reflected from (as determined by which trees' bounding boxes they fall into). Points
    not classified as "High Vegetation" (e.g. "Ground" points) are labeled as background.

    Returns two numpy arrays, coord_array which records the X and Y pixel coordinates of each pixel
    containing at least one LiDAR point, and label_array, which contains labels for each tree that
    pixel belongs to or 0 if it doesn't belong to any tree. (If a pixel contains LiDAR points possibly
    reflected from multiple trees, each tree label is included; if it contains LiDAR points reflected
    from both the ground and a tree, only the tree label is included.)

    params:
        lidar_folder (str): Path to folder containing laz files
        rgb_folder (str): Path to folder containing tiff files
        filename (str): laz / tiff filename minus the extension
        boxes (np.array): T x 4 array of [x0,y0,x1,y1] bounding boxes, where T is the
                          total number of boxes (trees).
        classifications (dict): Dictionary of LiDAR classifications. Must include 
                                "High Vegetation" as a key, and a corresponding
                                integer label. Other keys are optional.
        img_size (tuple): RGB image size in pixels (H x W)

    returns:
        coord_array: 1 x P x 2 array of X,Y coordinates in pixel space, where P is the
                     number of pixels with LiDAR points.
        label_array: T x P array of 1s and 0s, where T is the number of trees and P is
                     the number of pixels with LiDAR points. For each pixel p in each
                     tree t, p==1 if the corresponding pixel belongs to that tree, and 
                     p==0 if it is a background pixel or belongs to another tree. If
                     there are no boxes to label individual trees, T = 1 and all pixels
                     are 1 if they are classified as "High Vegetation" and 0 otherwise.
    '''

    # Read Lidar file
    las = laspy.read(os.path.join(lidar_folder, f'{filename}.laz'))

    # LiDAR to DataFrame
    points = np.vstack((las.x, las.y, las.classification)).transpose()
    df = pd.DataFrame(points, columns=['x', 'y', 'classification'])
    df['classification'] == classifications['High Vegetation']

    # Align Lidar coordinates with tiff indices
    with rasterio.open(os.path.join(rgb_folder, filename+'.tif')) as rast_img:
        left, bottom, right, top = rast_img.bounds
        df = df[df['x'].between(left, right) & df['y'].between(bottom, top)]
        # Get pixel coordinates for each point.
        pixels = [rast_img.index(x,y) for x,y in zip(df['x'], df['y'])]
    df[['y_bin','x_bin']] = pixels
    df = df.astype({'x_bin':'int', 'y_bin':'int', 'classification':'int'})

    # Re-index points on the right-most and bottom-most edges of the image
    df[['x_bin','y_bin']] = df[['x_bin','y_bin']].clip([0,0],[img_size[1]-1,img_size[0]-1])

    # If boxes provided, use these to label Lidar data by individual trees
    if boxes is not None:
        df = df.groupby(by=['x_bin','y_bin'])['classification'].max()
        coord_array = df.reset_index()[['x_bin','y_bin']].to_numpy()[np.newaxis,:]
        label_array = np.zeros((len(boxes), coord_array.shape[1]), dtype='int')
        for i, box in enumerate(boxes):
            s = pd.Series(data=0, index=df.index)
            xmin, xmax = box[0], box[2]
            ymin, ymax = box[1], box[3]
            s.loc[xmin:xmax, ymin:ymax] = df.loc[xmin:xmax, ymin:ymax]
            label_array[i] = s

    # Otherwise, label all "High Vegetation" points 1.
    else:
        df = df.groupby(by=['x_bin','y_bin'])['classification'].max().reset_index()
        coord_array = df[['x_bin','y_bin']].to_numpy()[np.newaxis,:]
        label_array = df['classification'].to_numpy(dtype='int')[np.newaxis,:]

    return coord_array, label_array


def lidar_filter(boxes, coordinates, labels, threshold=0.2):
    '''
    Filter bounding boxes based on the percentage of labeled pixels within their area
    that are labeled as "Tree" vs "Non-Tree". Note that this percentage is calculated 
    only from the labeled pixels, meaning pixels with neither a "Tree" or "Non-Tree" 
    label are not included in the calculation. Boxes with lower than the threshold of
    "Tree" pixels (or else no labeled pixels) are removed. Returns an index for those
    boxes passing the threshold.

    params:
      boxes (ndarray): N x 4 array of bounding boxes, where N is the number of boxes
      coordinates (ndarray): 1 x P x 2 array of X,Y coordinates, where P is the number
                             of points.
      labels (ndarray): N x P array of binary labels, with 1 being a tree point inside the
                        Nth box and 0 being a non-tree point inside or outside the box
      threshold (float): Percent of points inside the box that should be marked as 1 or
                         "tree" in order to keep the box and labels as legitimate

    returns:
      idx (ndarray): Indices for the remaining boxes and labels after filtering
    '''
    index = pd.MultiIndex.from_arrays([coordinates[0,:,0], coordinates[0,:,1]])
    df = pd.DataFrame(data=labels.T, index=index)
    idx = []
    for i,box in enumerate(boxes):
        xmin, xmax = box[0], box[2]
        ymin, ymax = box[1], box[3]
        box_points = df[i].loc[xmin:xmax, ymin:ymax]
        if len(box_points) > 0 and box_points.sum() / len(box_points) > threshold:
            idx.append(i)

    return np.array(idx, dtype='int')


def sample_points(coordinates, labels, pos_samples, neg_samples, 
                  distance_weight=True, neg_sample_spread=1, seed=None):
    '''
    Sample from pixels with LiDAR points. Can specify the number of positive and negative samples 
    taken. If points are labeled collectively, will sample pos_samples and neg_samples from the
    entire image. If points are labeled individually, will sample pos_samples and neg_samples per
    tree. For individually labeled points, can choose to sample points uniformly or weighted by
    distance from the center of the tree. Weights will prioritize positive and negative points
    near the edges of trees to better delineate them.

    params:
        coordinates (ndarray): 1 x N x 2 array of X,Y coordinates, where N is the total number of points
        labels (ndarray): T x N array of binary labels, where T is the number of labeled trees
                          (1 for collective labels)
        pos_samples (int): The number of positive samples to draw, either in total for collectively
                           labeled points or per tree for individually labeled points
        neg_samples (int): The number of negative samples to draw, either in total for collectively
                           labeled points or per tree for individually labeled points
        distance_weight (bool): If True, give higher weight to points near the edges of trees; may
                                produce undesirable results with collectively labeled points, suggest
                                setting to False in this case
        neg_sample_spread (int): When using distance weights, use this number to adjust how spread out
                                 negative samples are from the tree; higher values indicate higher spread,
                                 while lower values indicate tighter clustering around trees
        seed (int): Random seed, set to an integer for reproducible results

    returns:
        sample_coordinates: T x (pos_samples + neg_samples) x 2 array of X,Y coordinates; if collectively
                            labeled points then T = 1
        sample_labels: T x (pos_samples + neg_samples) array of binary labels, with 1 for trees and 0 for
                       background; if collectively labeled points then T = 1    
    '''
    rng = np.random.default_rng(seed)
    sample_coordinates = []
    sample_labels = []


    # For each individual tree (or all trees if collectively labeled):
    for tree in labels:

        # Get the indices of the positive and negative points for that tree
        pos_indices = tree.nonzero()[0]
        neg_indices = (tree==0).nonzero()[0]

        # Get the coordinates for the positive and negative points
        pos_coords = coordinates[0,pos_indices]
        neg_coords = coordinates[0,neg_indices]

        # If there are fewer positive points than pos_samples, supplement with additional negative samples
        if len(pos_indices) >= pos_samples:
            num_pos = pos_samples
        else:
            num_pos = len(pos_indices)
        num_neg = pos_samples - num_pos + neg_samples

        # Distance weighting is only possible with > 0 positive points
        if distance_weight is True and num_pos > 0:
            # Find the center pixel of the positive points
            pos_center = pos_coords.mean(axis=0).reshape(1,-1)

            # Find the distances of all positive and negative points from the positive center
            pos_distances = euclidean_distances(pos_coords, pos_center)[:,0]
            neg_distances = euclidean_distances(neg_coords, pos_center)[:,0]

            # Only sample negative points that are at least as far from the tree centroid as the farthest positive point
            min_distance = pos_distances.max()
            neg_distances[neg_distances < min_distance] = np.inf

            # Use distances to calculate sampling probabilities
            # (Boundary points between tree vs background or tree vs tree have higher probability of being sampled)
            if num_pos > 1:
                pos_probs = pos_distances / sum(pos_distances)
            else:
                pos_probs = [1.0]

            # Use neg_sample_spread to further weight spread of negative points
            spread = 1 / neg_sample_spread
            neg_exp = np.exp(-spread*neg_distances)
            neg_probs = neg_exp / sum(neg_exp)

            # Select a random subset of positive indices
            pos_indices = rng.choice(pos_indices, num_pos, replace=False, p=pos_probs, shuffle=False)
            neg_indices = rng.choice(neg_indices, num_neg, replace=False, p=neg_probs, shuffle=False)

        else:
            pos_indices = rng.choice(pos_indices, num_pos, replace=False, shuffle=False)
            neg_indices = rng.choice(neg_indices, num_neg, replace=False, shuffle=False)

        # Combine positive and negative indices, use them to sample coordinates for given tree
        indices = np.concatenate([pos_indices, neg_indices])
        sample_coordinates.append(coordinates[0, indices])

        # Create new labels based on how many positive and negative samples were taken
        sample_labels.append(np.concatenate([np.ones(pos_indices.shape), np.zeros(neg_indices.shape)]))

    # Stack coordinates and labels, then return both arrays
    sample_coordinates = np.stack(sample_coordinates)
    sample_labels = np.stack(sample_labels)

    return sample_coordinates, sample_labels