"""
Definition of the architecture of the network used to predict the steering of the car in each frame
"""
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D, Dropout
from keras.models import Sequential


class DriveNeuralNetwork:
    def __init__(self, input_shape, train_generator, n_train, validation_generator, n_validation, n_epochs, batch_size):
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.train_generator = train_generator
        self.len_training = n_train
        self.validation_generator = validation_generator
        self.len_validation = n_validation
        self.epochs = n_epochs

    def train(self, nn_type='simple'):
        """
        
        :param nn_type: The type of the network to train 
        :return: the history of the execution
        """
        models = {
            'simple': lambda: simple_nn(self.input_shape),
            'nvidia': lambda: nvidia_nn(self.input_shape),
            'simplified': lambda: nvidia_simplified(self.input_shape),
            'alvinn': lambda: alvinn(self.input_shape)
        }
        model = models.get(nn_type, simple_nn(self.input_shape))()

        model.compile(optimizer='adam', loss='mean_squared_error')
        history = model.fit_generator(self.train_generator,
                                      samples_per_epoch=self.len_training * 2,
                                      validation_data=self.validation_generator,
                                      nb_val_samples=self.len_validation,
                                      nb_epoch=self.epochs)
        # TODO Provide the directory to save the models
        model.save('./models/{}.h5'.format(nn_type))
        return history


def simple_nn(input_shape):
    """
    Simple architecture used to test the wiring of the different components
    :return: a model
    """
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=input_shape, name='input'))
    model.add(Dense(1))
    return model


def nvidia_nn(input_shape):
    """
    Creates the architecture of the model defined by NVidia in their paper https://arxiv.org/pdf/1604.07316.pdf
    :return: a model
    """
    model = Sequential()
    # TODO NVidia nn uses 66x200x3
    # Normalization
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=input_shape, name='input'))
    # Convolution 24@31x98.  2×2 stride and a 5×5 kernel
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
    # Convolution 36@14x47. 2x2 stride and a 5x5 kernel
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
    # Convolution 48@5x22. 2x2 stride and a 5x5 kernel
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
    # Convolution 64@3x20. non-stride and a 3x3 kernel
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    # Convolution 64@1x18. non-stride and a 3x3 kernel
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    # Add Flatten layer to
    model.add(Flatten())
    # FC 1164
    model.add(Dense(1164, activation='relu'))
    model.add(Dropout(.5))
    # FC 100
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(.5))
    # FC 50
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(.5))
    # FC 10
    model.add(Dense(10, activation='relu'))
    # Output
    model.add(Dense(1, name='output'))
    return model


def nvidia_simplified(input_shape):
    model = Sequential()
    # Normalization
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=input_shape, name='input'))

    model.add(Convolution2D(16, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(32, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1))
    return model


def alvinn(input_shape):
    """
    This architecture is extracted from ALVINN experiments in the late '80s
    Note that there are some differences:
    * the input size in ALVINN is 30x32)
    * Also there is a normalization layer
    * The activation of the fully connected layer is RELU
    * The output is a single node (instead of the 30 nodes used by ALVINN to classify the steering values)
    :return: a model representing the ALVINN architecture
    """
    model = Sequential()
    # Normalization
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=input_shape, name='input'))
    model.add(Dense(5, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(30, activation='relu'))
    model.add(Flatten())
    model.add(Dense(1))
    return model
