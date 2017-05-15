import cv2
import glob

import matplotlib.image as mpimg
import numpy as np


def calibrate_camera(is_test, img_dir, image_logger=None):
    """
    Calibrate the camera by using chessboard corners.
    We need at least 10 images for camera calibration
    See here for more http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
    :param is_test: Whether we are in test mode or not
    :param img_dir: Directory where the images to perform camera calibration are
    :param image_logger: an ImageLogger instance
    :return: camera matrix, distortion coefficients
    """
    # These magic numbers are obtained from observing the images provided for calibration
    nx = 9
    ny = 6
    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
    # Get the first image and see what happens
    objpoints = []
    imgpoints = []
    n_corners_found = 0
    show_sample = is_test
    for f in glob.glob('{}/*.jpg'.format(img_dir)):
        img = mpimg.imread(f)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        if ret:
            n_corners_found += 1
            imgpoints.append(corners)
            objpoints.append(objp)

            if is_test and show_sample:
                img_corners = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
                if image_logger is not None:
                    image_logger.show_and_save_image(img_corners, 'chessboard_corners')
                show_sample = False

    print('Number of corners founds {}'.format(n_corners_found))
    # given object points, image points, and the shape of the grayscale image:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return mtx, dist


def get_perspective_matrix(corners_src, corners_dst):
    """
    Generates the 3x3 transformation matrix using the points passed as parameters
    :param corners_src: 
    :param corners_dst: 
    :return: 
    """
    M = cv2.getPerspectiveTransform(np.float32(corners_src), np.float32(corners_dst))
    Minv = cv2.getPerspectiveTransform(np.float32(corners_dst), np.float32(corners_src))
    return M, Minv


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    persp_args = {
        'corners_src': [[200, 700], [1070, 700], [750, 490], [530, 490]],
        'corners_dst': [[325, 700], [975, 700], [975, 0], [325, 0]]
    }
    img = mpimg.imread('./test_images/straight_lines1.jpg')
    M, Minv = get_perspective_matrix(persp_args['corners_src'], persp_args['corners_dst'])
    lines_img = np.zeros_like(img)
    dst_img = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
    lines_dst_img = np.zeros_like(img)

    plt.subplot(121)
    plt.imshow(img)
    xyList = [(x, y) for x, y in persp_args['corners_src']]
    p = Polygon(xyList, alpha=0.2)
    plt.gca().add_artist(p)
    plt.title('Input')

    plt.subplot(122)
    plt.imshow(dst_img)
    xyList = [(x, y) for x, y in persp_args['corners_dst']]
    p = Polygon(xyList, alpha=0.2)
    plt.gca().add_artist(p)

    plt.title('Output')
    plt.show()
