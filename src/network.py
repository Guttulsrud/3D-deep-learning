import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from config import config
from src.orthogonal_regularizer import OrthogonalRegularizer

"""
PointNet consists of two core components. The primary MLP network, and the transformer
net (T-net). The T-net aims to learn an affine transformation matrix by its own mini
network. The T-net is used twice. The first time to transform the input features (n, 3)
into a canonical representation. The second is an affine transformation for alignment in
feature space (n, 3). As per the original paper we constrain the transformation to be
close to an orthogonal matrix (i.e. ||X*X^T - I|| = 0).
"""


def get_point_net_model(hp):
    """
    The main network can be then implemented in the same manner where the t-net mini models
    can be dropped in a layers in the graph. Here we replicate the network architecture
    published in the original paper but with half the number of weights at each layer as we
    are using the smaller 10 class ModelNet dataset.
    """
    number_of_points = config['number_of_points']
    number_of_classes = config['number_of_classes']

    inputs = keras.Input(shape=(number_of_points, 3))
    x = get_t_net(inputs, 3)
    x = conv_bn(x, 32, hp)
    x = conv_bn(x, 32, hp)
    x = get_t_net(x, 32, hp)
    x = conv_bn(x, 32, hp)
    x = conv_bn(x, 64, hp)
    x = conv_bn(x, 512, hp)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 256, hp)
    x = layers.Dropout(hp.Choice('dropout', values=config['hpo']['dropout1']))(x)
    x = dense_bn(x, 128, hp)
    x = layers.Dropout(hp.Choice('dropout', values=config['hpo']['dropout2']))(x)

    outputs = layers.Dense(number_of_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="pointnet")
    model.summary()

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=config['network']['learning_rate']),
        metrics=["sparse_categorical_accuracy"],
    )

    return model


def get_t_net(inputs, num_features, hp=None):
    num_features = num_features * 2 if hp else num_features
    # Initialise bias as the identity matrix
    bias = keras.initializers.Constant(np.eye(num_features).flatten())
    reg = OrthogonalRegularizer(num_features)

    x = conv_bn(inputs, 32, hp)
    x = conv_bn(x, 64, hp)
    x = conv_bn(x, 512, hp)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 256, hp)
    x = dense_bn(x, 128, hp)
    x = layers.Dense(num_features * num_features,
                     kernel_initializer="zeros",
                     bias_initializer=bias,
                     activity_regularizer=reg,
                     )(x)
    feat_T = layers.Reshape((num_features, num_features))(x)

    print(feat_T.shape, inputs.shape)
    # Apply affine transformation to input features
    return layers.Dot(axes=(2, 1))([inputs, feat_T])


def conv_bn(x, filters, hp=None):
    if hp:
        filters = hp.Int('units', min_value=filters * 2, max_value=filters * 4, step=32)

    x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
    x = layers.BatchNormalization(momentum=0.0)(x)

    activation = hp.Choice('activation', values=config['hpo']['activation']) if hp else 'relu'
    return layers.Activation(activation)(x)


def dense_bn(x, filters, hp=None):
    if hp:
        filters = hp.Int('units', min_value=filters * 2, max_value=filters * 4, step=32)

    x = layers.Dense(filters)(x)
    x = layers.BatchNormalization(momentum=0.0)(x)

    activation = hp.Choice('activation', values=config['hpo']['activation']) if hp else 'relu'
    return layers.Activation(activation)(x)

# Tensorboard
# HPO Stuff
# metrics?
# sparse_categorical_accuracy
