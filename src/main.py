import datetime

import tensorflow as tf
import matplotlib
import keras_tuner as kt

from config import config
from src.data_loader import get_dataset
from src.evaluation import show_performance
from src.network import get_point_net_model
from src.shapenet_dataloader import ShapeNetDataLoader
from src.utils import save_results

tf.random.set_seed(config['random_seed'])
matplotlib.use('TkAgg')

if __name__ == "__main__":
    train_dataset, test_dataset, CLASS_MAP = get_dataset()

    log_dir = "logs/" + datetime.datetime.now().strftime("%m%d-%H%M")

    hist_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        embeddings_freq=1,
        write_graph=True,
        update_freq='batch')

    tuner = kt.BayesianOptimization(get_point_net_model,
                                    objective='sparse_categorical_accuracy', )

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='sparse_categorical_accuracy', patience=config['patience'])

    tuner.search(train_dataset, epochs=50, callbacks=[stop_early, hist_callback])

    best_hps = tuner.get_best_hyperparameters(num_trials=config['trials'])[0]

    model = tuner.hypermodel.build(best_hps)
    history = model.fit(train_dataset, epochs=50)

    val_acc_per_epoch = history.history['sparse_categorical_accuracy']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print('Best epoch: %d' % (best_epoch,))

    hypermodel = tuner.hypermodel.build(best_hps)

    # Retrain the model
    hypermodel.fit(train_dataset, epochs=best_epoch)

    eval_result = hypermodel.evaluate(test_dataset)
    print("[test loss, test accuracy]:", eval_result)
