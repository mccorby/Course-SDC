import math
import matplotlib.pyplot as plt


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
    columns = len(imgs) + 2
    plt.subplot(1, columns, 1)
    plt.imshow(original)
    plt.subplot(1, columns, 2)
    plt.imshow(grayscale, cmap='gray')
    for i in range(0, len(imgs)):
        plt.subplot(1, columns, i + 3)
        plt.imshow(imgs[i], cmap='gray')
        plt.axis('off')
    plt.show()


def show_histogram(labels, number_classes):
    """
    Shows the number of samples per traffic sign
    :param labels: 
    :param number_classes: 
    :return: 
    """
    plt.hist(labels, number_classes)
    plt.show()


def show_images(images, cmap='gray'):
    columns = len(images)
    plt.figure(figsize=(12, 12))
    for i in range(0, columns):
        plt.subplot(1, columns, i + 1)
        plt.imshow(images[i], cmap=cmap)
        plt.axis('off')
    plt.show()

