"""
The trainer acts as the controller
"""
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import data_processor
import models


def train(data_root_dir, log_filename, input_shape, epochs=3, batch_size=128, model_type='simple'):
    processor = data_processor.DataProcessor(data_root_dir, input_shape)
    dataset = data_processor.load_data(data_root_dir, log_filename)
    train_dataset, valid_dataset = train_test_split(dataset, test_size=0.2)
    train_generator = processor.generator(train_dataset, batch_size=batch_size)
    val_generator = processor.generator(valid_dataset, batch_size=batch_size)
    model = models.DriveNeuralNetwork((66, 200, 3), train_generator, len(train_dataset), val_generator, len(valid_dataset),
                                      epochs, batch_size)
    return model.train(model_type)


def show_loss(history):
    print(history.history.keys())
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('MSE loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

if __name__ == '__main__':
    EPOCHS = 10
    BATCH_SIZE = 64
    MODEL = 'nvidia'
    INPUT_SHAPE = (66, 200, 3)
    loss_history = train('./data', 'driving_log.csv', INPUT_SHAPE, EPOCHS, BATCH_SIZE, MODEL)
    show_loss(loss_history)
