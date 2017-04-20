import csv
import glob
import pickle
import os
import numpy as np
import cv2
import random

from cnn import LeNet
import tensorflow as tf

from sklearn.utils import shuffle

import data_augmenter
import data_explorer

SAVE_FILENAME = './alternative'
DATA_DIR = './traffic-signs-data/'
PROCESS_TRAINING = False


with open('signnames.csv') as sign_names_file:
    reader = csv.reader(sign_names_file)
    next(reader, None)
    sign_names = {int(row[0]): row[1] for row in reader}


training_file = os.path.join(DATA_DIR, 'train.p')
validation_file = os.path.join(DATA_DIR, 'valid.p')
testing_file = os.path.join(DATA_DIR, 'test.p')
with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

# Data exploration and visualization
n_classes = len(np.unique(y_train))
messages = data_explorer.summary_data(X_train, X_valid, X_test, n_classes)
[print(msg) for msg in messages]

data_explorer.visualize_dataset(X_train, y_train, n_classes, sign_names)


# Preprocessing
def normalize(img):
    return ((img - 128.) / 128.).astype(np.float32)


def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def histogram(img):
    return cv2.equalizeHist(img)


def preprocess(img):
    result = grayscale(img)
    result = histogram(result)
    result = normalize(result)
    return result

# Visualize how the preprocessing affects an image
img_index = random.randint(0, len(X_train) - 1)
random_img = X_train[img_index]
gray_random_img = grayscale(random_img)
hist_random_img = histogram(gray_random_img)
data_explorer.show_preprocessed_images(random_img, gray_random_img, hist_random_img)

# Show the number of samples per class
data_explorer.show_histogram(y_train, n_classes)

# Preprocess data
X_train = [preprocess(x) for x in X_train]
X_train = np.array(X_train)
X_train = X_train[..., np.newaxis]

X_valid = [preprocess(x) for x in X_valid]
X_valid = np.array(X_valid)
X_valid = X_valid[..., np.newaxis]

X_test = [preprocess(x) for x in X_test]
X_test = np.array(X_test)
X_test = X_test[..., np.newaxis]

X_train, y_train = shuffle(X_train, y_train)

print(X_train.shape)


# Augment data
random_augmented = data_augmenter.augment_image(random_img)
data_explorer.show_images(random_augmented, title='Augmented Images')

data_augmented, labels_augmented = data_augmenter.augment_dataset(X_train, y_train)
print('data_augmented size {}'.format(len(data_augmented)))
data_augmented = np.array(data_augmented)[..., np.newaxis]
labels_augmented = np.array(labels_augmented)
X_train = np.append(X_train, data_augmented, axis=0)
y_train = np.append(y_train, labels_augmented, axis=0)

print('Data augmentation. Augmented size {}. Correct labels? {}'.format(len(data_augmented),
                                                                        len(data_augmented) == len(labels_augmented)))

# ### Model Architecture

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, None)
one_hot_y = tf.one_hot(y, n_classes)
keep_prob = tf.placeholder(tf.float32)

rate = 0.001

logits = LeNet(x, n_classes, use_dropout=True, keep_prob=keep_prob)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()


def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


# ### Train, Validate and Test the Model

EPOCHS = 10
BATCH_SIZE = 128

# Train the model
# Calculate and report the accuracy on the training and validation set.

if PROCESS_TRAINING:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        num_examples = len(X_train)

        print("Training...")
        print()
        for i in range(EPOCHS):
            X_train, y_train = shuffle(X_train, y_train)
            for offset in range(0, num_examples, BATCH_SIZE):
                end = offset + BATCH_SIZE
                batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})

            train_accuracy = evaluate(X_train, y_train)
            validation_accuracy = evaluate(X_valid, y_valid)
            print("EPOCH {} ...".format(i + 1))
            print("Train Accuracy = {:.3f}".format(train_accuracy))
            print("Validation Accuracy = {:.3f}".format(validation_accuracy))
            print()

        test_accuracy = evaluate(X_test, y_test)
        print("Test Accuracy = {:.3f}".format(test_accuracy))
        saver.save(sess, SAVE_FILENAME)
        print("Model saved")


# Test a Model on New Images
predictions_img_filter = './unseen_images/*.png'

# Load unseen images
images_path = [file for file in glob.glob(predictions_img_filter)]
original_images = []
for image_path in images_path:
    print(image_path)
    image = cv2.imread(image_path)
    # Images coming in BGR color format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_images.append(image)

processed_original_images = [preprocess(img) for img in original_images]
processed_original_images = np.array(processed_original_images)[..., np.newaxis]

# Run the predictor
with tf.Session() as sess:
    if not PROCESS_TRAINING:
        saver.restore(sess, SAVE_FILENAME)
    print("Model restored")
    # Feed image into feed_dict
    predictor = tf.argmax(logits, 1)
    predictions = sess.run(predictor, feed_dict={x: processed_original_images, keep_prob: 1.0})


[print('{} - {}'.format(pred, sign_names[pred])) for pred in predictions]
data_explorer.show_predictions(original_images, sign_names, predictions)

# Calculate the accuracy for the unseen images
# Output Top 5 Softmax Probabilities For Each Image
TOP_K = 5
saver = tf.train.Saver()
with tf.Session() as sess:
    if not PROCESS_TRAINING:
        saver.restore(sess, SAVE_FILENAME)
    predictions = sess.run(tf.nn.softmax(logits), feed_dict={x: processed_original_images, keep_prob: 1.0})
    top_k = sess.run(tf.nn.top_k(predictions, TOP_K))


data_explorer.show_images_top_predictions(original_images, sign_names, top_k)


# def output_feature_map(sess, img, tf_activation, activation_min=-1, activation_max=-1, plt_num=1):
#     activation = tf_activation.eval(session=sess, feed_dict={x: img})
#
#     data_explorer.show_feature_maps(activation, activation_min, activation_max, plt_num)
#
#
# tf.reset_default_graph()
# with tf.variable_scope('conv1', reuse=True):
#     layer1 = tf.Variable(tf.truncated_normal(shape=(8, 8, 1, 20), mean=0., stddev=0.1), name='weights')
#
# with tf.variable_scope('conv2'):
#     layer2 = tf.Variable(tf.truncated_normal(shape=(5, 5, 20, 16), mean=0., stddev=0.1), name='weights')
#
# saver = tf.train.Saver()
# with tf.Session() as sess:
#     if not PROCESS_TRAINING:
#         saver.restore(sess, SAVE_FILENAME)
#     output_feature_map(sess, [original_images[0]], layer1)
#     output_feature_map(sess, [original_images[0]], layer2)
