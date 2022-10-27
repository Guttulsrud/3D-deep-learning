import tensorflow as tf
import matplotlib

from config import config
from src.data_loader import get_dataset
from src.evaluation import show_performance
from src.network import get_point_net_model

tf.random.set_seed(config['random_seed'])
matplotlib.use('TkAgg')

if __name__ == "__main__":
    train_dataset, test_dataset, CLASS_MAP = get_dataset()

    model = get_point_net_model()
    model.fit(train_dataset, epochs=config['epochs'], validation_data=test_dataset)

    show_performance(test_dataset, model, CLASS_MAP)
