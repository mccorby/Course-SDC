import cv2

from moviepy.video.io.VideoFileClip import VideoFileClip
from scipy.ndimage.measurements import label
from sklearn.externals import joblib

from feature_extraction import bin_spatial, color_hist, get_hog_features, convert_to_colorspace
from heatmap import update_heat, Heatmap, draw_labeled_bboxes
from sliding_window import search_windows, draw_boxes, slide_window
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_ubyte


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, hog_params, spatial_params, color_params):
    hist_bins = color_params['nbins']
    spatial_size = spatial_params['size']
    pix_per_cell = hog_params['pix_per_cell']
    cell_per_block = hog_params['cell_per_block']
    orient = hog_params['orient']

    bboxes = []

    img = img.astype(np.float32) / 255

    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = convert_to_colorspace(img_tosearch, color_space='YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block ** 2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            _, _, _, _, hist_features = color_hist(subimg, nbins=hist_bins)
            features = np.hstack((spatial_features, hist_features, hog_features))
            test_features = X_scaler.transform(features.reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                top_left = (xbox_left, ytop_draw + ystart)
                bottom_right = (xbox_left + win_draw, ytop_draw + win_draw + ystart)
                bboxes.append((top_left, bottom_right))

    return bboxes


def pipeline(frame):
    heat = np.zeros_like(frame[:, :, 0]).astype(np.float)
    bboxes = find_cars(frame, ystart, ystop, scale, model, scaler, hog_params, spatial_params, color_params)
    draw_img = np.copy(frame)
    # for box in bboxes:
    #     cv2.rectangle(draw_img, box[0], box[1], (0, 0, 255), 6)
    heatmap = update_heat(draw_img, heat, bboxes)
    result = cv2.addWeighted(draw_img, 1, heatmap, 0.4, 0)
    return result


def pipeline_copy(frame):
    bboxes = []
    current_frame_heatmap = Heatmap()
    for i in range(len(windows_params['scales'])):
        # xstart = windows_params['x_limits'][i][0]
        # xstop = windows_params['x_limits'][i][1]
        ystart = windows_params['y_limits'][i][0]
        ystop = windows_params['y_limits'][i][1]
        scale = windows_params['scales'][i]
        current_bboxes = find_cars(frame, ystart, ystop, scale, model, scaler, hog_params, spatial_params, color_params)
        bboxes.extend(current_bboxes)

    current_frame_heatmap.add_heat(bboxes)
    heatmap.update(current_frame_heatmap)
    heatmap.apply_threshold(5)
    labels = label(heatmap.heatmap)
    draw_img = draw_labeled_bboxes(np.copy(frame), labels)
    # im_heatmap = heatmap.get_rgb_image()
    # new_hm = np.zeros_like(draw_img)
    # new_hm[:,:,0] = im_heatmap
    # new_hm[:,:,1] = im_heatmap
    # new_hm[:,:,2] = im_heatmap
    # print(new_hm.dtype)
    # result = cv2.addWeighted(draw_img, 1, new_hm, 0.4, 0)
    return draw_img


if __name__ == '__main__':
    # image = mpimg.imread('./test_images/test1.jpg')
    # Use this if working with JPG.
    # If you extracted training data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    # draw_image = np.copy(image)
    # image = image.astype(np.float32) / 255

    y_start_stop = [375, 700]  # Min and max in y to search in slide_window() 375
    window_sizes = [(192, 192), (96, 96), (48, 48)]
    # TODO Check if model exists
    model = joblib.load('./model.pkl')
    scaler = joblib.load('./scaler.pkl')
    color_space = 'YCrCb'
    color_params = {'nbins': 32, 'bins_range': (0, 256)}
    spatial_params = {'size': (32, 32)}
    hog_params = {'orient': 9, 'pix_per_cell': 8, 'cell_per_block': 2, 'channel': 'ALL'}
    xy_overlap = (0.5, 0.5)

    ystart = 400
    ystop = 656
    scale = 1.25

    # for window_size in window_sizes:
    #     windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
    #                            xy_window=window_size, xy_overlap=xy_overlap)
    #
    #     hot_windows = search_windows(image, windows, model, scaler, color_space,
    #                                  color_params, spatial_params, hog_params)
    #
    #     draw_image = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)
    #
    # plt.title('Overlap {}.{}'.format(xy_overlap[0], xy_overlap[1]))
    # plt.imshow(draw_image)
    # plt.show()

    # bboxes = find_cars(image, ystart, ystop, scale, model, scaler, hog_params, spatial_params, color_params)
    # draw_img = np.copy(image)
    # for box in bboxes:
    #     cv2.rectangle(draw_img, box[0], box[1], (0, 0, 255), 6)
    #
    # plt.imshow(draw_img)
    # plt.show()
    heatmap = Heatmap()
    windows_params = {'sizes': [(64, 64), (96, 96), (128, 128)],
                      'y_limits': [[400, 500], [380, 600], [500, 650]],
                      'x_limits': [[0, 1280], [0, 1280], [0, 1280]],
                      'scales': [1.25, 1.5]}
    clip2 = VideoFileClip('./project_video.mp4')
    vid_clip = clip2.fl_image(pipeline_copy)
    vid_clip.write_videofile('./test_video_output.mp4', audio=False)
