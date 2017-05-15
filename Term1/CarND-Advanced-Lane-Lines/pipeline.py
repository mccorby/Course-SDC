import argparse
import cv2
import glob
from collections import deque

import matplotlib.pyplot as plt
import numpy as np

from thresholds import hls_filter, lab_filter
from utils import load_image, ImageLogger, calculate_center
from window_search import find_window_by_histogram, measure_curvature, fit_with_previous_frame


class LaneFinderPipeline:
    """
    This class represents the pipeline applied to each frame to obtained the location of the lanes
    """

    def __init__(self, args=None, mtx=None, dist_coeff=None, image_logger=None, thresholds=None, M=None, Minv=None):
        self.image_logger = image_logger
        self.dist_coeff = dist_coeff
        self.mtx = mtx
        self.args = args
        self.thresholds = thresholds
        self.M = M
        self.Minv = Minv
        self.current_state = None
        self.right_line = Line()
        self.left_line = Line()

    def execute(self, image):
        """
        Applies the pipeline of operations to the input image
        :param image: the original image
        :return: the image after being processed by the pipeline
        """
        undist = self.undistort(image)
        result = self.threshold_image(undist, self.thresholds['ksize'],
                                      self.thresholds['sobel'],
                                      self.thresholds['magnitude'],
                                      self.thresholds['direction'],
                                      self.thresholds['saturation'],
                                      self.thresholds['lightness'],
                                      self.thresholds['blue-yellow'])
        warped = self.warp(result)
        if self.args.is_test:
            self.image_logger.save_image(warped, 'warped_image.png')
        ploty, left_fit, right_fit, left_fitx, right_fitx = self.get_line_fit(warped)
        self.left_line.update(left_fit)
        self.right_line.update(right_fit)
        left_rad, right_rad = measure_curvature(ploty, self.args.is_test)
        result = self.draw_final_image(image, warped, undist, ploty, left_fitx, right_fitx, self.Minv, left_rad,
                                       right_rad)
        return result

    def undistort(self, image):
        """
        Undistort an image using previously calculated matrix and distortion coefficientes
        :param image: the original image
        :return: the image undistorted
        """
        dst = cv2.undistort(image, self.mtx, self.dist_coeff, None)

        if self.args.is_test:
            self.image_logger.save_image(dst, 'undistorted')
            images = [[{'title': 'Original', 'data': image},
                       {'title': 'Undistorted', 'data': dst}]]
            self.image_logger.plot_results(images)
        return dst

    # TODO Refactor this method
    def threshold_image(self, image, ksize, sobel_thresh, mag_thresh, dir_thresh, s_thresh, l_thresh, b_thresh):
        """
        Transform the image by applying thresholds to different attributes
        :param image: the original image
        :param ksize: Sobel kernel size
        :param sobel_thresh: Threshold to apply to the Sobel operator
        :param mag_thresh: Threshold to apply to the magnitude calculation of Sobel operator
        :param dir_thresh: Threshold to apply to the direction of the gradient
        :param s_thresh: Threshold to apply to the Saturation channel of the image (in HLS color space)
        :param l_thresh: Threshold to apply to the Lightness channel of the image (in HLS color space)
        :param b_thresh: Threshold to apply to the blue-yellow channel (in Lab color space)
        :return: a binary representation of the image after applying the filters
        """
        # Note: Magnitude and direction thresholds were not needed for the project. Probably they are for the challenges
        # For Sobel, the light channel will be used
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS).astype(np.float)
        l_channel = hls[:, :, 1]
        # Sobel x
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=ksize)  # Take the derivative in x
        abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

        # Threshold x gradient
        gradient_binary = np.zeros_like(scaled_sobel)
        gradient_binary[(scaled_sobel >= sobel_thresh[0]) & (scaled_sobel <= sobel_thresh[1])] = 1

        s_binary, l_binary = hls_filter(image, s_thresh, l_thresh)
        l_color_channel = lab_filter(image, b_thresh)
        binary = np.zeros_like(gradient_binary)
        binary[((l_binary == 1) | (l_color_channel == 1))] = 1
        binary = 255 * np.dstack((binary, binary, binary)).astype('uint8')
        images = [
            [{'title': 'Original', 'data': image},
             {'title': 'Full Combined', 'data': binary}
             ]
        ]
        title = 'Kernel = {}; sobel = {}, mag = {}, dir = {}, s_filter = {}, l_filter = {}' \
            .format(ksize, sobel_thresh, mag_thresh, dir_thresh, s_thresh, l_thresh)
        if self.args.is_test:
            self.image_logger.plot_results(images, title)

        return binary

    def warp(self, img):
        return cv2.warpPerspective(img, self.M, (img.shape[1], img.shape[0]))

    def window_search(self, img, method='histogram'):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if method == 'histogram':
            return find_window_by_histogram(gray, is_test=self.args.is_test)
            # TODO If using convolution, it is returning the centroids positions
            # if method == 'convolution':
            #     window_width = 50
            #     window_height = 80  # Break image into 9 vertical layers since image height is 720
            #     margin = 100  # How much to slide left and right for searching
            #
            #     return find_window_centroids_convolutions(img, window_width, window_height, margin)

    def draw_final_image(self, image, warped, undist, ploty, left_fitx, right_fitx, Minv, left_rad, right_rad):
        """
        Draw the different outputs into the original image using the params passed
        The image is modified by adding a polygon that fits between the lanes and information about the curvature and
        where the center of the lane is
        :param image: the original image
        :param warped:
        :param undist:
        :param ploty:
        :param left_fitx:
        :param right_fitx:
        :param Minv:
        :param left_rad:
        :param right_rad:
        :return:
        """
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(gray).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

        font = cv2.FONT_HERSHEY_SIMPLEX
        off_center = calculate_center(left_fitx, right_fitx, image.shape)
        direction_str = 'left' if off_center < 0 else 'right'
        center_str = '{:.2f} m of center {}'.format(abs(off_center), direction_str)
        cv2.putText(result, center_str, (430, 630), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
        if left_rad and right_rad:
            curvature = 0.5 * (round(right_rad / 1000, 1) + round(left_rad / 1000, 1))
        else:
            curvature = 0
        str2 = 'Radius of curvature: {} km'.format(curvature)
        cv2.putText(result, str2, (430, 670), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

        if self.args.is_test:
            plt.imshow(result)
            plt.show()

        return result

    def get_line_fit(self, warped):
        # We've got a previous detection... use it
        if self.right_line.detected and self.left_line.detected:
            ploty, left_fit, right_fit, left_fitx, right_fitx = fit_with_previous_frame(warped,
                                                                                        self.left_line.best_fit,
                                                                                        self.right_line.best_fit)
        else:
            print('Doing new window search')
            ploty, left_fit, right_fit, left_fitx, right_fitx = self.window_search(warped)
        return ploty, left_fit, right_fit, left_fitx, right_fitx


# Define a class to receive the characteristics of each line detection
class Line:
    def __init__(self, iterations=5):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = deque(maxlen=iterations)
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None

    def update(self, fit):
        """
        Determine if the new fit is good enough and decide whether this line is a good detection or not
        :param fit: the new fit for the line
        """
        if fit is None:
            self.detected = False
        else:
            if self.best_fit is None:
                self._accept_fit(fit)
            else:
                # We have a previous best fit. Compare it with the incoming one
                if self._sanity_check(fit):
                    self._accept_fit(fit)
                else:
                    self.detected = False

    def _sanity_check(self, fit):
        # TODO Add more sanity checks
        # Check curvature
        # Check horizontal separation
        # Check parallelism
        diffs = abs(fit - self.best_fit)
        if diffs[0] > 0.1 or diffs[1] > 1.0 or diffs[2] > 100.:
            return False
        return True

    def _accept_fit(self, fit):
        """
        Accept the incoming fit for the line and update the values for the best fit by averaging the contents of the
        deque
        :param fit: the new fit
        """
        self.detected = True
        self.current_fit.append(fit)
        self.best_fit = np.average(self.current_fit, axis=0)


if __name__ == '__main__':
    ksize = 3
    sobel = (20, 255)  # was (20, 120)
    mag_thresh = (30, 150)
    dir_thresh = (0.7, 1.3)
    s_filter = (120, 255)
    l_filter = (40, 255)

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', help='Is test', action='store_false', dest='is_test', default=True)
    args = parser.parse_args()

    image_logger = ImageLogger(None)
    pipeline = LaneFinderPipeline(image_logger=image_logger, args=args)
    [pipeline.threshold_image(load_image(filename), ksize, sobel, mag_thresh, dir_thresh, s_filter, l_filter) for
     filename in glob.glob('./test_images/*')]
