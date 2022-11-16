import tensorflow as tf


def jitter(points, label):
    # jitter points
    #todo: make augmentations optional in the config file and put params in config file
    points += tf.random.uniform(points.shape, -0.005, 0.005, dtype=tf.float64)
    return points, label
