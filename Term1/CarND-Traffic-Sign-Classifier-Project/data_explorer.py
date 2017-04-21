import math
import matplotlib.pyplot as plt
import numpy as np


def summary_data(train, validation, test, n_classes):
    messages = []
    # What's the shape of an traffic sign image?
    image_shape = train[0].shape

    messages.append('Number of training examples = {}'.format(len(train)))
    messages.append('Number of validation examples = {}'.format(len(validation)))
    messages.append('Number of testing examples = {}'.format(len(test)))
    messages.append('Image data shape = {}'.format(image_shape))
    messages.append('Number of classes = {}'.format(n_classes))
    return messages


def visualize_dataset(dataset, labels, n_classes, sign_names):
    columns = 4
    rows = math.ceil(n_classes / columns)
    plt.figure(figsize=(18, 18))
    for i in range(0, n_classes):
        plt.subplot(rows, columns, i + 1)
        x_selected = dataset[labels == i]
        plt.imshow(x_selected[0, :, :, :])
        plt.title('{} - {}'.format(i, sign_names[i]))
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def show_preprocessed_images(original, grayscale, *imgs):
    """
    Show an image and the result of its preprocessing
    :param grayscale: 
    :param original: 
    :param imgs: 
    :return: 
    """
    plt.figure(figsize=(16, 16))
    plt.suptitle('Preprocessed Images')
    columns = len(imgs) + 2
    plt.subplot(1, columns, 1)
    plt.imshow(original)
    plt.subplot(1, columns, 2)
    plt.imshow(grayscale, cmap='gray')
    for i in range(0, len(imgs)):
        plt.subplot(1, columns, i + 3)
        plt.axis('off')
        plt.imshow(imgs[i], cmap='gray')
    plt.show()


def show_histogram(labels, number_classes):
    """
    Shows the number of samples per traffic sign
    :param labels: 
    :param number_classes: 
    :return: 
    """
    plt.title('Samples per class')
    for label_set in labels:
        plt.hist(label_set, number_classes)
    plt.show()


def show_images(images, title, cmap='gray'):
    columns = len(images)
    plt.figure(figsize=(12, 8))
    plt.suptitle(title)
    for i in range(0, columns):
        plt.subplot(1, columns, i + 1)
        plt.axis('off')
        plt.imshow(images[i], cmap=cmap)
    plt.show()


def show_predictions(images, sign_names, predictions):
    columns = 4
    rows = math.ceil(len(images) / columns)
    plt.figure(figsize=(12, 8))
    plt.suptitle('Predictions')
    for i in range(len(images)):
        plt.subplot(rows, columns, i + 1)
        plt.imshow(images[i])
        plt.title(sign_names[predictions[i]])
        plt.axis('off')
    plt.show()


def show_images_top_predictions(images, sign_names, top_values):
    plt.figure(figsize=(16, 21))
    for i in range(len(images)):
        plt.subplot(12, 2, 2*i+1)
        plt.imshow(images[i])
        plt.title(i)
        plt.axis('off')
        plt.subplot(12, 2, 2*i+2)
        plt.barh(np.arange(1, 6, 1), top_values.values[i, :])
        labs = [sign_names[j] for j in top_values.indices[i]]
        plt.yticks(np.arange(1, 6, 1), labs)
    plt.show()


def show_feature_maps(activation, plt_num, activation_min, activation_max):
    feature_maps = activation.shape[3]
    plt.figure(plt_num, figsize=(15, 15))
    for feature_map in range(feature_maps):
        plt.subplot(6, 8, feature_map + 1)  # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(feature_map))  # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0, :, :, feature_map], interpolation="nearest", vmin=activation_min,
                       vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0, :, :, feature_map], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min != -1:
            plt.imshow(activation[0, :, :, feature_map], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0, :, :, feature_map], interpolation="nearest", cmap="gray")
    
    plt.show()
