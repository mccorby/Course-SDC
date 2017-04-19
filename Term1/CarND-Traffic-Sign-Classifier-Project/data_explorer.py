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
