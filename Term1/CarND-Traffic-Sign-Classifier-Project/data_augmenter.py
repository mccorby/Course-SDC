import itertools
import random
import cv2


def apply_gaussian_blur(image):
    """
    Smooth image by applying Gaussian blur
    :param image: the original image
    :return: a new image 
    """
    return cv2.GaussianBlur(image, (5, 5), 0)


def increase_contrast(image):
    """
    Increase the intensity of the points of the image
    :param image: the original image
    :return: a new image with its points more intense
    """
    max_intensity = 255.
    return max_intensity * (image / max_intensity) ** 0.5


def rotate_image(img):
    """
    Rotate the image by a random angle
    :param img: the original image
    :return: a new image rotated some random angle
    """
    num_rows, num_cols = img.shape[:2]
    angle = random.randint(-5, 5)
    rotation_matrix = cv2.getRotationMatrix2D((num_cols / 2, num_rows / 2), angle, 1)
    return cv2.warpAffine(img, rotation_matrix, (num_cols, num_rows))


def augment_image(img):
    """
    Creates new images applying different modifications to the original image
    :param img: the original image
    :return: a set of new images
    """
    result = [rotate_image(img), apply_gaussian_blur(img)]
    return result


def augment_dataset(dataset, labels):
    """
    Creates new images from the original dataset
    :param dataset: the original dataset
    :param labels: the original labels
    :return: a new set of data and labels
    """
    augmented_data = []
    augmented_labels = []
    for i in range(0, len(dataset)):
        augmented_images = augment_image(dataset[i])
        augmented_data.extend(augmented_images)
        augmented_labels += len(augmented_images) * [labels[i]]
    return augmented_data, augmented_labels
