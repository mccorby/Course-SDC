import cv2

import numpy as np


def abs_sobel_thresh(sobel, thresh):
    """
    Applies Sobel x or y, then takes an absolute value and applies a threshold.

    :param sobel
    :param thresh: 
    :return: 
    """
    assert (len(thresh) == 2)
    # Calculate gradient

    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # Apply threshold
    mask = (scaled_sobel > thresh[0]) & (scaled_sobel < thresh[1])
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[mask] = 1
    return binary_output


def magnitude_thresh(sobel_values, mag_thresh):
    """
    Computes the magnitude of the gradient and applies a threshold
    :param sobel_values: 
    :param mag_thresh: 
    :return: 
    """
    assert (len(sobel_values) == 2)
    # Calculate the magnitude of the Sobel gradients
    abs_sobelxy = np.sqrt(np.square(sobel_values[0]) + np.square(sobel_values[1]))
    scaled_sobel = (255 * abs_sobelxy / np.max(abs_sobelxy))
    # Create a binary mask where mag thresholds are met
    mask = (scaled_sobel > mag_thresh[0]) & (scaled_sobel < mag_thresh[1])
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[mask] = 1
    return binary_output


def direction_thresh(sobel, dir_thresh):
    """
    Computes the direction of the gradient and applies a threshold.

    :param sobel: 
    :param dir_thresh: 
    :return: 
    """
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobel[0])
    abs_sobely = np.absolute(sobel[1])
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    dir_gradient = np.arctan2(abs_sobely, abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
    mask = (dir_gradient > dir_thresh[0]) & (dir_gradient < dir_thresh[1])
    binary_output = np.zeros_like(dir_gradient)
    binary_output[mask] = 1
    # 6) Return this mask as your binary_output image
    return binary_output


def gradient_threshold(image, ksize=3, sobel_thresh=(0, 255), mag_thresh=(0, 255), dir_thresh=(0, np.pi / 2)):
    """
    Apply gradient threshold to the image. This method applies Sobel, magnitude and direction thresholding to the image
    :param image: the original image
    :param ksize: the size of the Sobel kernel
    :param sobel_thresh: the thresholding to apply to the Sobel representation of the image
    :param mag_thresh: the thresholding to apply to the Sobel magnitude representation of the image
    :param dir_thresh: the thresholding to apply to the direction representation of the image
    :return: a binary representation of the image with the different filters applied
    """
    # If the image has already being preprocessed, keep it. Otherwise, convert to gray
    if len(image.shape) < 3:
        gray = image
    else:
        # All operations work with a grayscale image
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Apply each of the thresholding functions
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    gradx = abs_sobel_thresh(sobelx, thresh=sobel_thresh)
    grady = abs_sobel_thresh(sobely, thresh=sobel_thresh)
    mag_binary = magnitude_thresh((sobelx, sobely), mag_thresh=mag_thresh)
    dir_binary = direction_thresh((sobelx, sobely), dir_thresh=dir_thresh)
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    return combined, mag_binary, dir_binary


def hls_filter(image, s_thresh=(0, 255), l_thresh=(0, 255)):
    """
    Produce two binary representations using the saturation and lightness channels in a HLS color space of the image
    :param image: the original image
    :param s_thresh: the threshold to apply to the saturation channel
    :param l_thresh: the threshold to apply to the lightness channel
    :return: binary representations of the filtered image
    """
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    l_mask = (l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])
    l_binary_output = np.zeros_like(l_channel)
    l_binary_output[l_mask] = 1

    s_mask = (s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])
    s_binary_output = np.zeros_like(s_channel)
    s_binary_output[s_mask] = 1

    return s_binary_output, l_binary_output


def lab_filter(image, thresh=(0, 255)):
    """
    Filter by B channel in a Lab color space of image
    :param image: the original image
    :param thresh: the threshold to apply to each pixel
    :return: a binary representation of the image filtered by the threshold
    """
    # 1) Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
    lab_b = lab[:, :, 2]
    # don't normalize if there are no yellows in the image
    if np.max(lab_b) > 175:
        lab_b = lab_b * (255 / np.max(lab_b))
    # 2) Apply a threshold to the b channel
    binary_output = np.zeros_like(lab_b)
    binary_output[((lab_b > thresh[0]) & (lab_b <= thresh[1]))] = 1
    # 3) Return a binary image of threshold result
    return binary_output
