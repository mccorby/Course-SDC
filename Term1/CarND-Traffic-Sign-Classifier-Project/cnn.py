import tensorflow as tf
from tensorflow.contrib.layers import flatten


def LeNet(x, n_classes, use_dropout=False, keep_prob=0.7):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1

    # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    # Variables created here will be named "conv1/weights", "conv1/biases".
    with tf.variable_scope('conv1'):
        conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean=mu, stddev=sigma), name='weights')
        conv1_b = tf.Variable(tf.zeros(6), name='biases')
        conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

        # SOLUTION: Activation.
        conv1 = tf.nn.relu(conv1)

        if use_dropout:
            conv1 = tf.nn.dropout(conv1, keep_prob)

        # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    with tf.variable_scope('conv2'):
        # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
        conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma), name='weights')
        conv2_b = tf.Variable(tf.zeros(16), name='biases')
        conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

        # SOLUTION: Activation.
        conv2 = tf.nn.relu(conv2)

        if use_dropout:
            conv2 = tf.nn.dropout(conv2, keep_prob)

        # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    with tf.variable_scope('fc0'):
        tf.add_to_collection('weights', conv2)
        fc0 = flatten(conv2)

    with tf.variable_scope('fc1'):
        # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
        fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma), name='weights')
        fc1_b = tf.Variable(tf.zeros(120), name='biases')
        fc1 = tf.matmul(fc0, fc1_W) + fc1_b

        # SOLUTION: Activation.
        fc1 = tf.nn.relu(fc1)
        if use_dropout:
            fc1 = tf.nn.dropout(fc1, keep_prob)

    with tf.variable_scope('fc2'):
        # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
        fc2_W = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma), name='weights')
        fc2_b = tf.Variable(tf.zeros(84), name='biases')
        fc2 = tf.matmul(fc1, fc2_W) + fc2_b

        # SOLUTION: Activation.
        fc2 = tf.nn.relu(fc2, name='fc2')
        if use_dropout:
            fc2 = tf.nn.dropout(fc2, keep_prob)

    with tf.variable_scope('fc3'):
        # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 43.
        fc3_W = tf.Variable(tf.truncated_normal(shape=(84, n_classes), mean=mu, stddev=sigma), name='weights')
        fc3_b = tf.Variable(tf.zeros(n_classes), name='biases')
        logits = tf.matmul(fc2, fc3_W) + fc3_b

    tf.add_to_collection('logits', logits)
    return logits
