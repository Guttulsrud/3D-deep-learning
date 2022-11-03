import tensorflow as tf
import matplotlib
import keras_tuner as kt

from config import config
from src.data_loader import get_dataset
from src.evaluation import show_performance
from src.network import get_point_net_model

tf.random.set_seed(config['random_seed'])
matplotlib.use('TkAgg')

if __name__ == "__main__":
    train_dataset, test_dataset, CLASS_MAP = get_dataset()

    tuner = kt.BayesianOptimization(get_point_net_model,
                                    objective='val_accuracy')

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    tuner.search(train_dataset, epochs=50, validation_split=0.2, callbacks=[stop_early])

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"""
        The hyperparameter search is complete. The optimal number of units in the first densely-connected
        layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
        is {best_hps.get('learning_rate')}.
    """)
