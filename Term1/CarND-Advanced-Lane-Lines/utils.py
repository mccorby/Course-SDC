import cv2

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os


def load_image(file_path):
    """
    Loads an image from the file path
    :param file_path: the location of the image
    :return: the image
    """
    return mpimg.imread(file_path)


class ImageLogger:
    """
    Class used to log information about the images
    This class includes methods to show a grid of images, save images to output directory
    """

    def __init__(self, img_output_dir):
        self.img_output_dir = img_output_dir

    def save_image(self, img, filename):
        path_filename = os.path.join(self.img_output_dir, filename)
        mpimg.imsave('{}.png'.format(path_filename), img)

    def show_and_save_image(self, img, filename):
        self.save_image(img, filename)
        plt.imshow(img)
        plt.show()

    def plot_results(self, result, title=None):
        # Plot the result
        n_rows = len(result)
        n_cols = len(result[0])
        if title is not None:
            fig = plt.gcf()
            fig.suptitle(title, fontsize=14)

        for i, row in enumerate(result):
            for j, img in enumerate(row):
                ax = plt.subplot2grid((n_rows, n_cols), (i, j))
                plt.title(img['title'])
                ax.imshow(img['data'], cmap='gray')
        plt.show()


def project_lines_onto_image(image, undist, warped, Minv, left_fitx, right_fitx, ploty):
    """
    Display a polygon between the lanes represented by right_fitx and left_fitx
    :param image: the original image
    :param undist: the undistorted image
    :param warped: the warped image
    :param Minv: the inverse of the perspective Matrix 
    :param left_fitx: x values of the left lane
    :param right_fitx: x values of the right lane
    :param ploty: 
    :return: None
    """
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    new_warp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, new_warp, 0.3, 0)
    plt.imshow(result)


def calculate_center(rightx, leftx, image_shape):
    """
    Calculates the center point between a left and right given points
    :param rightx:
    :param leftx:
    :return: the center x point
    """
    xm_per_pix = 3.7/700 # meteres per pixel in x dimension
    lane_width = 3.7
    # off_center = -100 * round(0.5 * (rightx - lane_width / 2) + 0.5 * (abs(leftx) - lane_width / 2), 2)
    middle = (leftx[-1] +rightx[-1])//2
    veh_pos = image_shape[1]//2

    off_center = (veh_pos - middle)*xm_per_pix # Positive if on right, Negative on left
    return off_center


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img).astype(np.uint8)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, [vertices], ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
