from src.network.layers import conv_bn, dense_bn
from src.network.t_net import get_t_net
from tensorflow import keras
from tensorflow.keras import layers
from config import config

"""
PointNet consists of two core components. The primary MLP network, and the transformer
net (T-net). The T-net aims to learn an affine transformation matrix by its own mini
network. The T-net is used twice. The first time to transform the input features (n, 3)
into a canonical representation. The second is an affine transformation for alignment in
feature space (n, 3). As per the original paper we constrain the transformation to be
close to an orthogonal matrix (i.e. ||X*X^T - I|| = 0).
"""


def get_point_net_model():
    """
    The main network can be then implemented in the same manner where the t-net mini models
    can be dropped in a layers in the graph. Here we replicate the network architecture
    published in the original paper but with half the number of weights at each layer as we
    are using the smaller 10 class ModelNet dataset.
    """

    number_of_points = config['number_of_points']
    number_of_classes = config['number_of_classes']
    learning_rate = config['learning_rate']

    inputs = keras.Input(shape=(number_of_points, 3))

    x = get_t_net(inputs, 3)
    x = conv_bn(x, 32)
    x = conv_bn(x, 32)
    x = get_t_net(x, 32)
    x = conv_bn(x, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 256)
    x = layers.Dropout(0.3)(x)
    x = dense_bn(x, 128)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(number_of_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="pointnet")
    model.summary()

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=["sparse_categorical_accuracy"],
    )

    return model
