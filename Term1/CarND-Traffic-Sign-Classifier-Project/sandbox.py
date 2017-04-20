import itertools
import math


def augment_dataset(dataset, labels):
    pairs = zip(dataset, labels)
    zip_result = [(img + 10, label) for (img, label) in pairs]
    aug_data, aug_labels = zip(*zip_result)
    return aug_data, aug_labels


d = [1, 2, 3, 4, 5]
l = ['A', 'B', 'C', 'D', 'E']

aug_data, aug_labels = augment_dataset(d, l)
print(aug_data)
print(aug_labels)

augmented_labels = [label * 3 for label in l]
print(augmented_labels)


a_list = [[1, 1, 1], [2, 2, 2]]
list_of_list = [elem * 2 for elem in a_list]
print(list_of_list)

merged = list(itertools.chain(*list_of_list))
print(merged)

print(math.ceil(43 / 4))


def show_multiple(*args):
    print(len(args))
    print(args[0])

show_multiple(5, 4, 3, 2)

print('Printing slicing')
[print(x) for x in l[:3]]
