"""
This is the entry point in the project
"""
import argparse

import numpy as np
import os
from moviepy.editor import VideoFileClip

from calibration import calibrate_camera, get_perspective_matrix
from pipeline import LaneFinderPipeline
from utils import ImageLogger, load_image


def lane_finder(args, thresholds, persp_params):
    image_logger = ImageLogger(args.output_dir)
    if not os.path.isfile('{}.npz'.format(args.cam_calib_filename)):
        mtx, dist_coeff = calibrate_camera(args.is_test, args.camera_img_dir, image_logger)
        np.savez_compressed(args.cam_calib_filename, mtx=mtx, coeff=dist_coeff)
    else:
        print('Reading calibration data from file')
        cam_data = np.load('{}.npz'.format(args.cam_calib_filename))
        mtx = cam_data['mtx']
        dist_coeff = cam_data['coeff']

    M, Minv = get_perspective_matrix(persp_params['corners_src'], persp_params['corners_dst'])

    pipeline = LaneFinderPipeline(args, mtx, dist_coeff, image_logger, thresholds, M, Minv)
    if args.is_test:
        test_image = load_image(os.path.join(args.test_dir, 'test2.jpg'))
        result = pipeline.execute(test_image)
        image_logger.plot_results([[{'title': 'Pipelined Img', 'data': result}]])
        image_logger.save_image(result, 'binary_combo')
    else:
        print('Processing video...')
        clip2 = VideoFileClip(args.video_filename)
        vid_clip = clip2.fl_image(pipeline.execute)
        vid_clip.write_videofile(args.video_output_filename, audio=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', help='Is test', action='store_false', dest='is_test', default=True)
    parser.add_argument('-d', help='Input Images directory', dest='data_dir', type=str, default='none')
    parser.add_argument('-c', help='Calibration images dir', dest='camera_img_dir', type=str, default='./camera_cal')
    parser.add_argument('-o', help='Output directory', dest='output_dir', type=str, default='./output_images')
    parser.add_argument('-t_dir', help='Test images directory', dest='test_dir', type=str, default='./test_images')
    parser.add_argument('-v', help='Input video file', dest='video_filename', type=str, default='./project_video.mp4')
    parser.add_argument('-cam', help='Calibration file', dest='cam_calib_filename', type=str,
                        default='./support_files/camera_cal_data')
    parser.add_argument('-ov', help='Video output filename', dest='video_output_filename', type=str,
                        default='video_result.mp4')

    args = parser.parse_args()

    # These values were obtained by testing differente combinations
    thresholds = {'ksize': 3,
                  'sobel': (20, 255),
                  'magnitude': (30, 150),
                  'direction': (0.7, 1.3),
                  'saturation': (120, 255),
                  'lightness': (220, 255),
                  'blue-yellow': (190, 255)}

    # These values were obtained by direct observation
    persp_args = {
        'corners_src': [[200, 700], [1070, 700], [750, 490], [530, 490]],
        'corners_dst': [[325, 700], [975, 700], [975, 0], [325, 0]]
    }

    lane_finder(args, thresholds, persp_args)
