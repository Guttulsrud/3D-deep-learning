import tensorflow as tf
import matplotlib
import keras_tuner as kt
from keras.callbacks import LearningRateScheduler

from config import config
from src.data_loader import get_dataset
from src.network import get_point_net_model
from src.utils import initialize_callbacks, save_results, decay_schedule

tf.random.set_seed(config['random_seed'])
# matplotlib.use('TkAgg')

if __name__ == "__main__":
    info = f'{"no" if not config["hpo"]["enabled"] else ""} HPO. Vanilla PointNet jittering'
    # This makes it easier to look at run logs!
    print(info)

    train_dataset, test_dataset, CLASS_MAP = get_dataset(load_file='ModelNetX.json')

    hpo_enabled = config['hpo']['enabled']
    if hpo_enabled:
        print('Run with HPO')
        # Find optimal hyperparameters
        tuner = kt.BayesianOptimization(get_point_net_model,
                                        objective='sparse_categorical_accuracy',
                                        max_trials=config['trials'])
        print('Run search')
        tuner.search(train_dataset, epochs=config['epochs'], callbacks=initialize_callbacks(), validation_data=test_dataset)
        print('Find best HP')

        best_hps = tuner.get_best_hyperparameters()[0]

        # Train the model with the optimal hyperparameters
        print('Train the model with the optimal hyperparameters')
        model = tuner.hypermodel.build(best_hps)
        history = model.fit(train_dataset, epochs=config['epochs'], verbose=2)

        val_acc_per_epoch = history.history['sparse_categorical_accuracy']
        best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
        print('Best epoch: %d' % (best_epoch,))

        # Retrain the model
        print('Retrain model')
        hypermodel = tuner.hypermodel.build(best_hps)

        hypermodel.fit(train_dataset, epochs=best_epoch, verbose=2)

        eval_result = hypermodel.evaluate(test_dataset)
        print("[test loss, test accuracy]:", eval_result)
    else:
        print('Run without HPO')
        model = get_point_net_model()
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='sparse_categorical_accuracy',
                                                      patience=config['patience'])
        lr_scheduler = LearningRateScheduler(decay_schedule)
        history = model.fit(train_dataset, epochs=config['epochs'], callbacks=initialize_callbacks())

        val_acc_per_epoch = history.history['sparse_categorical_accuracy']
        best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
        print('Best epoch: %d' % (best_epoch,))
