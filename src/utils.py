import datetime
import json
import os
from config import config
from datetime import datetime
import tensorflow as tf
from keras.callbacks import LearningRateScheduler


def save_results(results: dict):
    if not os.path.exists('results'):
        os.mkdir('results')

    dataset = config['data_dir'].split('data/')[1]

    output = {
        'config': config,
        'results': results
    }
    now = datetime.now().strftime("%Y-%m-%d %H.%M")
    path = f'results/{now} -- {dataset}.json'
    with open(path, "w") as outfile:
        outfile.write(json.dumps(output))


def decay_schedule(epoch, lr):
    # decay by 0.1 every 5 epochs; use `% 1` to decay after each epoch
    if (epoch % 20 == 0) and (epoch != 0):
        lr = lr * 0.5
    return lr


def initialize_callbacks():
    global hist_callback, stop_early, lr_scheduler
    log_dir = "logs/" + datetime.now().strftime("%m%d-%H%M")
    hist_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        embeddings_freq=1,
        write_graph=True,
        update_freq='batch')
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='sparse_categorical_accuracy', patience=config['patience'])
    lr_scheduler = LearningRateScheduler(decay_schedule)

    return [hist_callback, stop_early, lr_scheduler]
