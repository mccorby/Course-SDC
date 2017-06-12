# To tune your SVM vehicle detection model, you can use one of scikit-learn's parameter tuning algorithms.
#
# When tuning SVM, remember that you can only tune the C parameter with a linear kernel. For a non-linear kernel,
# you can tune C and gamma.
import pathlib
import time

import matplotlib.image as mpimg
import numpy as np
from sklearn import svm, model_selection
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from data_preprocessing import get_list_of_images
from feature_extraction import extract_features


# Two steps:
#   First, tune the classifier
#   Second: train the model with the selected parameters


def tune_classifier(X_train, y_train):
    parameters = {'C': np.arange(0.025, 1.25, 0.25)}
    svr = svm.SVC(kernel='linear')
    clf = model_selection.GridSearchCV(svr, parameters)
    clf.fit(X_train, y_train)
    print(clf.best_params_)


def create_datasets(car_filenames, noncar_filenames):
    """
    Create the train and test datasets
    :param car_filenames:
    :param noncar_filenames:
    :return:
    """
    X, labels, scaler = extract_features_dataset(car_filenames, noncar_filenames)
    scaled_X = scaler.transform(X)
    # Shuffle and split
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, labels, test_size=0.20, random_state=rand_state)

    return X_train, X_test, y_train, y_test, scaler


def train_classifier(X_train, y_train, X_test, y_test):
    # Check the training time for the SVC
    t = time.time()
    print('Feature vector length:', len(X_train[0]))
    svc = LinearSVC(loss='hinge')
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    return svc


def extract_features_dataset(car_filenames, noncars_filenames):
    color_space = 'YCrCb'
    color_params = {'nbins': 32, 'bins_range': (0, 256)}
    spatial_params = {'size': (32, 32)}
    hog_params = {'orient': 9, 'pix_per_cell': 8, 'cell_per_block': 2, 'channel': 'ALL'}
    car_features = []
    for image_filename in car_filenames:
        image = mpimg.imread(image_filename)
        car_features.append(extract_features(image, color_space, color_params, spatial_params, hog_params))

    noncar_features = []
    for image_filename in noncars_filenames:
        image = mpimg.imread(image_filename)
        noncar_features.append(extract_features(image, color_space, color_params, spatial_params, hog_params))

    X = np.vstack((np.array(car_features), np.array(noncar_features))).astype(np.float64)
    print('Vector X to fit in scaler {}'.format(X.shape))
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)

    labels = np.hstack((np.ones(len(car_features)), np.zeros(len(noncar_features))))
    return X, labels, X_scaler


def predict(model, X_test, y_test):
    # Check the prediction time for a single sample
    t = time.time()
    n_predict = 10
    print('My SVC predicts: ', model.predict(X_test[0:n_predict]))
    print('For these', n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2 - t, 5), 'Seconds to predict', n_predict, 'labels with SVC')


if __name__ == '__main__':
    perform_tuning = False
    cars = get_list_of_images('./train_images/vehicles')
    notcars = get_list_of_images('./train_images/non-vehicles')

    X_train, X_test, y_train, y_test, scaler = create_datasets(cars, notcars)
    joblib.dump(scaler, './scaler.pkl')
    if perform_tuning:
        tune_classifier(X_train, y_train)
    else:
        path = pathlib.Path('./model.pkl')
        if path.is_file():
            model = joblib.load('./model.pkl')
        else:
            model = train_classifier(X_train, y_train, X_test, y_test)
            joblib.dump(model, './model.pkl')
    predict(model, X_test, y_test)


