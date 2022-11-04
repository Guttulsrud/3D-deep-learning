import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow_graphics.datasets.shapenet import Shapenet
from config import config
from PointWOLF import PointWOLF

args = {'w_num_anchor':1,
        'w_sample_type':'fps',
        'w_sigma':0.5,
        'w_R_range':10,
        'w_S_range':3,
        'w_T_range':0.25}

class ShapeNetDataLoader:

    def __init__(self, **kwargs):

        for k, v in kwargs.items():
            if k in args.keys():
                args[k] = v
            else:
                raise Exception('Argument not in PointWOLF')

        self.pwolf = PointWOLF(args)
        print(args)

    def get_train_data(self):
        data_set = Shapenet.load(split='train', download_and_prepare_kwargs={
            'download_config': tfds.download.DownloadConfig(manual_dir=config['data_dir'])})
        data_set = data_set.map(lambda x: tf.numpy_function(func=self.pwolf.augment_parallel,
                                                            inp=[x['trimesh']['vertices'], x['label']],
                                                            Tout=[tf.float32, tf.int64]))
        return data_set

    def get_test_data(self):
        data_set = Shapenet.load(split='test', download_and_prepare_kwargs={
            'download_config': tfds.download.DownloadConfig(manual_dir=config['data_dir'])})
        data_set = data_set.map(lambda x: [x['trimesh']['vertices'], x['label']])
        return data_set

    def get_valid_data(self):
        data_set = Shapenet.load(split='validation', download_and_prepare_kwargs={
            'download_config': tfds.download.DownloadConfig(manual_dir=config['data_dir'])})
        data_set = data_set.map(lambda x: [x['trimesh']['vertices'], x['label']])
        return data_set

    def plot_pointcloud(self, points):
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(points[:, 0], points[:, 1], points[:, 2])
            ax.set_axis_off()
            plt.show()

if __name__ == "__main__":

    sn = ShapeNetDataLoader()
    print(ex := sn.get_train_data())
    print(sn.get_test_data())
    print(sn.get_valid_data())
    for i in ex.take(1):
        print(f'Label {i[1]}')
        sn.plot_pointcloud(i[0])
