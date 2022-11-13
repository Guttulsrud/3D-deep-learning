import tensorflow as tf
import matplotlib

from config import config
from src.data_loader import get_dataset
from src.evaluation import show_performance
from src.network import get_point_net_model
from src.shapenet_dataloader import ShapeNetDataLoader

tf.random.set_seed(config['random_seed'])
matplotlib.use('TkAgg')

if __name__ == "__main__":
    train_dataset, test_dataset, CLASS_MAP = get_dataset(load_file=True)
    # print(train_dataset)
    # dl = ShapeNetDataLoader()
    # train_dataset = dl.get_train_data()
    # test_dataset = dl.get_test_data()
    # print(train_dataset)
    model = get_point_net_model()
    model.fit(test_dataset, epochs=config['epochs'], validation_data=test_dataset)

    show_performance(test_dataset, model, CLASS_MAP)
