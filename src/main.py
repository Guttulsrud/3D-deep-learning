import tensorflow as tf
import matplotlib
import keras_tuner as kt

from config import config
from src.data_loader import get_dataset
from src.network import get_point_net_model
from src.utils import initialize_callbacks, save_results

tf.random.set_seed(config['random_seed'])
matplotlib.use('TkAgg')

if __name__ == "__main__":
    train_dataset, test_dataset, CLASS_MAP = get_dataset(load_file=config['data_dir'], pointwolf=False)

    # Find optimal hyperparameters
    tuner = kt.BayesianOptimization(get_point_net_model, objective='sparse_categorical_accuracy')

    tuner.search(train_dataset, epochs=config['epochs'], callbacks=initialize_callbacks())

    best_hps = tuner.get_best_hyperparameters(num_trials=config['trials'])[0]

    # Train the model with the optimal hyperparameters
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(train_dataset, epochs=config['epochs'])

    val_acc_per_epoch = history.history['sparse_categorical_accuracy']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print('Best epoch: %d' % (best_epoch,))

    # Retrain the model
    hypermodel = tuner.hypermodel.build(best_hps)

    hypermodel.fit(train_dataset, epochs=best_epoch)

    eval_result = hypermodel.evaluate(test_dataset)
    print("[test loss, test accuracy]:", eval_result)
