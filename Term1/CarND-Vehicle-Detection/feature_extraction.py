import cv2

import numpy as np
from skimage.feature import hog


def color_hist(img, nbins=32, bins_range=None):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Generating bin centers
    # We can compute the bin centers from the bin edges.
    # Each of the histograms in this case have the same bins,
    # so I'll just use the channel1_hist bin edges
    bin_edges = channel1_hist[1]
    bin_centers = (bin_edges[1:] + bin_edges[0:len(bin_edges) - 1]) / 2
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return channel1_hist, channel2_hist, channel3_hist, bin_centers, hist_features


def bin_spatial(img, size=(32, 32)):
    """
    This function is used as part of the subsampling for HOG optimization
    :param img:
    :param size:
    :return:
    """
    color1 = cv2.resize(img[:, :, 0], size).ravel()
    color2 = cv2.resize(img[:, :, 1], size).ravel()
    color3 = cv2.resize(img[:, :, 2], size).ravel()
    return np.hstack((color1, color2, color3))


# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    result = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                 cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
                 visualise=True, feature_vector=feature_vec, block_norm='L2-Hys')

    if vis == True:
        return result[0], result[1]
    else:
        return result[0]


def convert_to_colorspace(img, color_space='RGB'):
    """
    Convert an image to a color space representation
    :param img: the origianl image
    :param color_space: the color space defined as a 3-letter flag
    :return:
    """
    # Pass the color_space flag as 3-letter all caps string like 'HSV' or 'LUV' etc.
    # Convert image to new color space (if specified)
    feature_image = None
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    else:
        feature_image = np.copy(img)
    return feature_image


def extract_features(img, color_space='RGB', color_params=None, spatial_params=None, hog_params=None):
    feature_image = convert_to_colorspace(img, color_space)
    hist_features = spatial_features = hog_features = []
    if color_params is not None:
        rhist, ghist, bhist, _, hist_features = color_hist(feature_image, nbins=color_params['nbins'],
                                                           bins_range=color_params['bins_range'])

    if spatial_params is not None:
        # Apply bin_spatial() to get spatial color features
        spatial_features = bin_spatial(feature_image, size=spatial_params['size'])

    if hog_params is not None:
        if hog_params['channel'] == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                ch_hog_features = get_hog_features(feature_image[:, :, channel],
                                                   orient=hog_params['orient'], pix_per_cell=hog_params['pix_per_cell'],
                                                   cell_per_block=hog_params['cell_per_block'], feature_vec=False)
                hog_features.append(ch_hog_features)
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_params['channel']], orient=hog_params['orient'],
                                            pix_per_cell=hog_params['pix_per_cell'],
                                            cell_per_block=hog_params['cell_per_block'], vis=False, feature_vec=False)
        hog_features = np.ravel(hog_features)
    return np.hstack((spatial_features, hist_features, hog_features))
