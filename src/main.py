import medmnist
import numpy as np
from numpy import load
import k3d
from k3d.colormaps import paraview_color_maps
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from medmnist import INFO, Evaluator
import plotly.graph_objects as go

data = load('../data/synapsemnist3d.npz')
lst = data.files

train_images, val_images, test_images = [], [], []
train_labels, test_labels, val_labels = [], [], []

for item in lst:
    match item:
        case 'train_images':
            train_images = data[item]
        case 'val_images':
            val_images = data[item]
        case 'test_images':
            test_images = data[item]
        case 'train_labels':
            train_labels = data[item]
        case 'val_labels':
            val_labels = data[item]
        case 'test_labels':
            test_labels = data[item]

for image, label in zip(train_images, train_labels):

    plt.imshow(image)
    plt.show()
    print('e+')

    exit()
