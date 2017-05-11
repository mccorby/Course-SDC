"""
Loads the data to feed the network
"""
import csv
import cv2

import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.utils import shuffle

CROP_SHAPE = (60, -25)


def load_data(root_dir, csv_filename):
    """
    :param root_dir: 
    :param csv_filename: 
    
    :return:  
    """
    image_files = []
    with open(os.path.join(root_dir, csv_filename)) as csv_file:
        csv_reader = csv.DictReader(csv_file)
        # TODO Find a better data loader. Panda?
        for line in csv_reader:
            image_files.append({'center': line['center'].strip(),
                                'left': line['left'].strip(),
                                'right': line['right'].strip(),
                                'steering': float(line['steering'].strip())
                                })

    return image_files


def preprocess(image, input_shape):
    image = crop(image)
    image = resize(image, input_shape)
    return image


def crop(image):
    return image[CROP_SHAPE[0]:CROP_SHAPE[1], :, :]


def resize(image, input_shape):
    """
    Resize the image to the input shape used by the network model
    """
    return cv2.resize(image, (input_shape[1], input_shape[0]), cv2.INTER_AREA)


class DataProcessor:
    def __init__(self, root_dir, input_shape):
        self.input_shape = input_shape
        self.root_dir = root_dir

    def generator(self, dataset, batch_size, training_mode=True):
        """
        Generator of processed images
        :return: a generator that yields batches of processed images
        """
        correction = .18  # Was .25 and .20

        cameras = ['left', 'center', 'right']
        cameras_steering_correction = [correction, 0., -correction]
        while True:

            dataset = shuffle(dataset)
            for offset in range(0, len(dataset), batch_size):
                images = []
                angles = []
                batch_images = dataset[offset: offset + batch_size]
                for batch_image in batch_images:
                    # Select a random camera from the current sample if in training mode
                    camera = np.random.randint(len(cameras)) if training_mode else 1
                    image = cv2.imread(os.path.join(self.root_dir, batch_image[cameras[camera]]))
                    # Modify the steering angle accordingly
                    angle = batch_image['steering'] + cameras_steering_correction[camera]
                    # Augmented images
                    if training_mode:
                        augmented_image, augmented_angle = random_augment(image.copy(), angle)
                        augmented_image = preprocess(augmented_image, self.input_shape)
                        images.append(augmented_image)
                        angles.append(augmented_angle)

                    image = preprocess(image, self.input_shape)
                    images.append(image)
                    angles.append(angle)

                processed_features = np.array(images)
                processed_labels = np.array(angles)

                yield shuffle(processed_features, processed_labels)


def flip_image(image, angle):
    return cv2.flip(image, 1), -angle


def random_augment(image, angle):
    image, angle = flip_image(image, angle)

    # TODO Add shadow, change brightness, vertical tilt
    return image, angle


def show_images(original, augmented, preprocessed, image_titles):
    images = [original, augmented, preprocessed]
    columns = len(images)
    plt.figure(figsize=(12, 8))
    plt.suptitle('Augmented images')
    for i in range(0, columns):
        plt.subplot(1, columns, i + 1)
        plt.axis('off')
        plt.title('{} - {}'.format(i, image_titles[i]))
        plt.imshow(images[i])
    plt.show()


if __name__ == '__main__':
    EPOCHS = 10
    BATCH_SIZE = 64
    MODEL = 'simplified'
    INPUT_SHAPE = (66, 200, 3)
    data_root_dir = './data'
    log_filename = 'driving_log.csv'
    processor = DataProcessor(data_root_dir, INPUT_SHAPE)
    dataset = load_data(data_root_dir, log_filename)

    original_img = cv2.imread(os.path.join(data_root_dir, dataset[0]['center']))
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    augmented_imgs, augmented_angle = random_augment(original_img, dataset[0]['steering'])
    original_img_processed = preprocess(original_img, INPUT_SHAPE)
    show_images(original_img, augmented_imgs, original_img_processed, ['original', 'flipped', 'preprocessed'])

