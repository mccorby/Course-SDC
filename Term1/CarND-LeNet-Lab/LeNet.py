import tensorflow as tf


class SimpleConvBlock:
    """
    SimpleConvBlock builds blocks of CNN formed by:
     * a convolution layer
     * an activation function
     * pooling layer
    """
    def __init__(self, name, input_data, input_shape, output_shape, pooling_stride, stride, padding):
        """
        :param name: Name of this block
        :param input_shape: The shape of the input in 3D
        :param output_shape: The shape of the output in 3D
        :param pooling_stride: The pooling shape defined as hxw
        :param stride: A single value as proposed by the literature. "A stride of 2"
        :param padding: A numerical value
        """
        self.name = name
        self.input_data = input_data
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.pooling_stride = pooling_stride
        self.stride = stride
        self.padding = padding

    def build_block(self):
        """
        Builds a block with the
        :return:
        """
        with tf.variable_scope(self.name):
            input_data = tf.placeholder(tf.float32, shape=self.input_shape, name="input")
            filter_shape = self._get_filter_shape()
            filter_weights = tf.Variable(tf.truncated_normal(filter_shape))
            biases = tf.Variable(tf.zeros(self.output_shape[2]))
            stride = [1, self.stride, self.stride, 1]
            padding = 'SAME'
            conv_layer = tf.nn.conv2d(input_data, filter_weights, stride, padding, name='conv_layer')
            conv_biases = tf.nn.bias_add(conv_layer, biases)
            output = tf.relu(conv_biases)
            # TODO This should be passed also as a parameter
            ksize = [1, 2, 2, 1]

            pooling = tf.nn.max_pool(output, ksize, self.pooling_stride, padding)

            return pooling

    def _get_filter_shape(self):
        # From the formula to obtain the output of the layer
        height = self.input_shape[0] + 2 * self.padding - (self.output_shape[0] - 1) / self.stride
        width = self.input_shape[1] + 2 * self.padding - (self.output_shape[1] - 1) / self.stride
        input_depth = self.input_shape[2]
        output_depth = self.output_shape[2]
        return list([height, width, input_depth, output_depth])


def preprocess():
    """
    An MNIST image is initially 784 features (1D).
    We reshape this to (28, 28, 1) (3D), normalize such that the values are between 0-1 instead of 0-255, and finally,
    pad the image with 0s such that the height and width are 32 (centers digit further).
    Thus, the input shape going into the first convolutional layer is 32x32x1.
    :return:
    """
    pass


def build_architecture(input_data):
    """

    :return:
    """
    # Convolution layer 1. The output shape should be 28x28x6.
    # Activation 1. Your choice of activation function.
    # Pooling layer 1. The output shape should be 14x14x6.
    input_shape = [32, 32, 1]
    output_shape = [14, 14, 6]
    pooling_stride = [1, 2, 2, 1]
    stride = 1
    padding = 0
    b1 = SimpleConvBlock('layer1', input_data, input_shape, output_shape, pooling_stride, stride, padding)
    output_b1 = b1.build_block()
    # Convolution layer 2. The output shape should be 10x10x16.
    # Activation 2. Your choice of activation function.
    # Pooling layer 2. The output shape should be 5x5x16.
    output_shape = [10, 10, 16]
    b2 = SimpleConvBlock('layer2', output_b1, b1.output_shape, output_shape, pooling_stride, stride, padding)

    tf.contrib.layers.flatten(b2)

    final_output = b2

    return final_output


def pipeline(x, labels, learning_rate):

    logits = build_architecture(x)
    loss_operation = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    