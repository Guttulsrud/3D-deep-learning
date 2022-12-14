import os
import glob
import trimesh
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import json
from PointWOLF import PointWOLF
from config import config
from src.augmentation import jitter, jitter_point_cloud, rotate_point_cloud


def dataset_sample():
    # Show a sample of the data
    mesh = trimesh.load(f'../{config["data_dir"]}/chair/train/chair_0001.off')
    mesh.show()
    points = mesh.sample(config['number_of_points'])
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    ax.set_axis_off()
    plt.show()


def parse_dataset(num_points, load_file=None):
    """
    Reads dataset (pre-build or original) and creates train and test lists with labels
    """
    if load_file:
        f = open(load_file)
        result = json.load(f)
        class_map = {int(k): v for k, v in result['class_map'].items()}

        return (
            np.array(result['train_points']),
            np.array(result['test_points']),
            np.array(result['train_labels']),
            np.array(result['test_labels']),
            class_map,
        )

    train_points = []
    train_labels = []
    test_points = []
    test_labels = []
    class_map = {}

    for i, folder in enumerate(os.listdir(config['data_dir'])):
        folder = f'../data/ModelNet40/{folder}'
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

    result = {
        'train_points': np.array(train_points).tolist(),
        'test_points': np.array(test_points).tolist(),
        'train_labels': np.array(train_labels).tolist(),
        'test_labels': np.array(test_labels).tolist(),
        'class_map': class_map
    }

    with open(f'ModelNetX.json', "w") as outfile:
        outfile.write(json.dumps(result))

    return (
        np.array(result['train_points']),
        np.array(result['test_points']),
        np.array(result['train_labels']),
        np.array(result['test_labels']),
        result['class_map'],
    )


args = {'w_num_anchor': 1,
        'w_sample_type': 'fps',
        'w_sigma': 0.5,
        'w_R_range': 10,
        'w_S_range': 3,
        'w_T_range': 0.25}


def set_shapes(img, label, img_shape):
    """
    Sets the shape of the model and label after losing it due to tf.numpy_function
    """
    img.set_shape(img_shape)
    label.set_shape([])
    return img, label


def get_dataset(load_file=None):
    """
    Creates tensors for train and test data with the augmentations, shuffling and batching
    """
    if load_file and not os.path.exists(load_file):
        raise Exception(f'{load_file} not found!')

    train_points, test_points, train_labels, test_labels, CLASS_MAP = parse_dataset(
        config['number_of_points'], load_file=load_file
    )

    train_dataset = tf.data.Dataset.from_tensor_slices((train_points, train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_points, test_labels))

    point_wolf_enabled = config['augmentations']['pointwolf']

    train_dataset = train_dataset.shuffle(len(train_points))

    train_dataset = train_dataset \
        .map(lambda x, y: tf.numpy_function(func=rotate_point_cloud, inp=[x, y], Tout=[tf.float64, tf.int32])) \
        .map(lambda x, y: set_shapes(x, y, [config['number_of_points'], 3]))

    train_dataset = train_dataset \
        .map(lambda x, y: tf.numpy_function(func=jitter_point_cloud, inp=[x, y], Tout=[tf.float64, tf.int32])) \
        .map(lambda x, y: set_shapes(x, y, [config['number_of_points'], 3]))

    if point_wolf_enabled:
        pwolf = PointWOLF(args)

        train_dataset = train_dataset \
            .map(lambda x, y: tf.numpy_function(func=pwolf.augment_parallel, inp=[x, y], Tout=[tf.float32, tf.int32])) \
            .map(lambda x, y: set_shapes(x, y, [config['number_of_points'], 3]))

    train_dataset = train_dataset.batch(config['batch_size'])

    test_dataset = test_dataset \
        .shuffle(len(test_points)) \
        .batch(config['batch_size'])

    return train_dataset, test_dataset, CLASS_MAP
