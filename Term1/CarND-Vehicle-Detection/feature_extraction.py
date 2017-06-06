import cv2

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hog

from data_preprocessing import get_list_of_images


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


# def bin_spatial(feature_image, size=(32, 32), ravel_features=True):
#     """
#     Compute color histogram features
#     :param ravel_features:
#     :param feature_image:
#     :param size:
#     :return:
#     """
#     features = cv2.resize(feature_image, size, interpolation=cv2.INTER_NEAREST)
#
#     if ravel_features:
#         # Use ravel() to create the feature vector
#         features = features.ravel()
#     # Return the feature vector
#     return features


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


if __name__ == '__main__':
    # image_filename = './test_images/test1.jpg'
    # image = cv2.imread(image_filename)
    # feature_image = convert_to_colorspace(image)
    # rh, gh, bh, bincen, feature_vec = color_hist(feature_image, nbins=32, bins_range=(0, 256))
    # show_histogram(rh, gh, bh, bincen)

    # Read a color image
    # img = cv2.imread(image_filename)
    # img_small_RGB = bin_spatial(img, ravel_features=False)  # OpenCV uses BGR, matplotlib likes RGB
    # img_small_rgb = img_small_RGB / 255.  # scaled to [0, 1], only for plotting
    #
    # color_spaces = ('RGB', 'HSV', 'LUV', 'HLS', 'YUV', 'YCrCb')
    # for color_space in color_spaces:
    #     # Convert subsampled image to desired color space(s)
    #     feature_image = convert_to_colorspace(img, color_space)
    #     img_small_color = bin_spatial(feature_image, ravel_features=False)
    #
    #     # Plot and show
    #     plot3d(img_small_color, img_small_rgb)
    #     plt.title(color_space)
    #     plt.show()

    # Generate a random index to look at a car image
    cars = get_list_of_images('./train_images/vehicles')
    noncars = get_list_of_images('./train_images/non-vehicles')
    ind = np.random.randint(0, len(cars))
    # Read in the image
    image = mpimg.imread(cars[ind])
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Define HOG parameters
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    # Call our function with vis=True to see an image output
    features, hog_image = get_hog_features(gray, orient,
                                           pix_per_cell, cell_per_block,
                                           vis=True, feature_vec=False)

    # Plot the examples
    # fig = plt.figure()
    # plt.subplot(121)
    # plt.imshow(image, cmap='gray')
    # plt.title('Example Car Image')
    # plt.subplot(122)
    # plt.imshow(hog_image, cmap='gray')
    # plt.title('HOG Visualization')
    # plt.show()

    # Test Extract Features
    # NOTE TODO Take care with the StandarScaler. It is using both cars and not-cars to compute
    # IF using only one of the sets (cars, for instance) the non-normalize graph is different
    # After testing the code provided in the lessons, this is all correct
    # Using a single image to test how the feature extract works with the scaler yields similar results

    X, y, scaler = extract_features_dataset(cars, noncars)
    scaled_X = scaler.transform(X)
    # Apply the scaler to X
    # Plot an example of raw and scaled features
    ind = 3
    image = mpimg.imread(cars[ind])
    fig = plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(image)
    plt.title('Original Image')
    plt.subplot(132)
    plt.plot(X[ind])
    plt.title('Raw Features')
    plt.subplot(133)
    plt.plot(scaled_X[ind])
    plt.title('Normalized Features')
    fig.tight_layout()
    plt.show()
