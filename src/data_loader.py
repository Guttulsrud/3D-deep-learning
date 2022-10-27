import os
import glob
import trimesh
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from src.augmentation import augment
from config import config


def dataset_sample():
    mesh = trimesh.load(f'../{config["data_dir"]}/chair/train/chair_0001.off')
    mesh.show()
    points = mesh.sample(config['number_of_points'])
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    ax.set_axis_off()
    plt.show()


def parse_dataset(num_points):
    train_points = []
    train_labels = []
    test_points = []
    test_labels = []
    class_map = {}
    folders = glob.glob(os.path.join(config['data_dir'], "[!README]*"))

    for i, folder in enumerate(folders):
        print("processing class: {}".format(os.path.basename(folder)))
        # store folder name with ID so we can retrieve later
        class_map[i] = folder.split("/")[-1]
        # gather all files
        train_files = glob.glob(os.path.join(folder, "train/*"))
        test_files = glob.glob(os.path.join(folder, "test/*"))

        for f in train_files:
            train_points.append(trimesh.load(f).sample(num_points))
            train_labels.append(i)

        for f in test_files:
            test_points.append(trimesh.load(f).sample(num_points))
            test_labels.append(i)

    return (
        np.array(train_points),
        np.array(test_points),
        np.array(train_labels),
        np.array(test_labels),
        class_map,
    )


def get_dataset():
    train_points, test_points, train_labels, test_labels, CLASS_MAP = parse_dataset(
        config['number_of_points']
    )

    train_dataset = tf.data.Dataset.from_tensor_slices((train_points, train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_points, test_labels))

    train_dataset = train_dataset\
        .shuffle(len(train_points))\
        .map(augment)\
        .batch(config['batch_size'])
    test_dataset = test_dataset\
        .shuffle(len(test_points))\
        .batch(config['batch_size'])

    return train_dataset, test_dataset, CLASS_MAP
