import cv2

import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
from scipy.ndimage.measurements import label
from sklearn.externals import joblib

from feature_extraction import bin_spatial, color_hist, get_hog_features, convert_to_colorspace
from heatmap import draw_labeled_bboxes, add_heat, apply_threshold


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
    global heatmap_avg
    heat = np.zeros_like(frame[:, :, 0]).astype(np.float)
    bboxes = []
    for i in range(len(windows_params['scales'])):
        ystart = windows_params['y_limits'][i][0]
        ystop = windows_params['y_limits'][i][1]
        scale = windows_params['scales'][i]
        current_bboxes = find_cars(frame, ystart, ystop, scale, model, scaler, hog_params, spatial_params, color_params)
        bboxes.extend(current_bboxes)

    heat = add_heat(heat, bboxes)
    current_heat = heatmap_avg + heat
    heatmap_avg = heat
    current_heat = apply_threshold(current_heat, 10)  # Apply threshold to help remove false positives
    # Visualize the heatmap when displaying
    heatmap = np.clip(current_heat, 0, 255)
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(frame), labels)
    return draw_img


if __name__ == '__main__':
    # TODO Check if model exists
    model = joblib.load('./model.pkl')
    scaler = joblib.load('./scaler.pkl')
    color_space = 'YCrCb'
    color_params = {'nbins': 32, 'bins_range': (0, 256)}
    spatial_params = {'size': (32, 32)}
    hog_params = {'orient': 9, 'pix_per_cell': 8, 'cell_per_block': 2, 'channel': 'ALL'}
    xy_overlap = (0.5, 0.5)

    windows_params = {'sizes': [(64, 64), (96, 96), (128, 128)],
                      'y_limits': [[400, 500], [380, 600], [500, 650]],
                      'x_limits': [[0, 1280], [0, 1280], [0, 1280]],
                      'scales': [1.25, 1.5]}
    heatmap_avg = np.zeros((720, 1280))  # Heatmap average
    clip2 = VideoFileClip('./project_video.mp4')
    vid_clip = clip2.fl_image(pipeline)
    vid_clip.write_videofile('./test_video_output.mp4', audio=False)
