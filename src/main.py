import tensorflow as tf
import matplotlib

from config import config
from src.data_loader import get_dataset
from src.evaluation import show_performance
from src.network import get_point_net_model
from src.shapenet_dataloader import ShapeNetDataLoader
from src.utils import save_results

tf.random.set_seed(config['random_seed'])
matplotlib.use('TkAgg')

if __name__ == "__main__":
    train_dataset, test_dataset, CLASS_MAP = get_dataset(load_file='ModelNet10.json')
    model = get_point_net_model()
    results = model.fit(train_dataset, epochs=config['epochs'], validation_data=test_dataset)
    save_results(results.history)
    show_performance(test_dataset, model, CLASS_MAP)
