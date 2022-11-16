import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from config import config
from src.orthogonal_regularizer import OrthogonalRegularizer


def get_point_net_model(hp=None):
    hpo_enabled = config['hpo']['enabled']
    number_of_points = config['number_of_points']
    number_of_classes = config['number_of_classes']

    inputs = keras.Input(shape=(number_of_points, 3))
    x = get_t_net(inputs, 3)
    x = conv_bn(x, 64)
    x = conv_bn(x, 64)
    x = get_t_net(x, 64)
    x = conv_bn(x, 64, '1', hp)
    x = conv_bn(x, 128, '2', hp)
    x = conv_bn(x, 1024, '3', hp)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 512, '4', hp)

    if hpo_enabled:
        x = layers.Dropout(hp.Choice('dropout_1', values=config['hpo']['dropout1']))(x)

    x = dense_bn(x, 256, '5', hp)

    if hpo_enabled:
        x = layers.Dropout(hp.Choice('dropout_2', values=config['hpo']['dropout2']))(x)
    else:
        x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(number_of_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="pointnet")
    # model.summary()

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=config['network']['learning_rate']),
        metrics=["sparse_categorical_accuracy"],
    )

    return model


def get_t_net(inputs, num_features):
    bias = keras.initializers.Constant(np.eye(num_features).flatten())
    reg = OrthogonalRegularizer(num_features)

    x = conv_bn(inputs, 64)
    x = conv_bn(x, 128)
    x = conv_bn(x, 1024)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 512)
    x = dense_bn(x, 256)
    x = layers.Dense(num_features * num_features,
                     kernel_initializer="zeros",
                     bias_initializer=bias,
                     activity_regularizer=reg,
                     )(x)
    feat_T = layers.Reshape((num_features, num_features))(x)

    # Apply affine transformation to input features
    return layers.Dot(axes=(2, 1))([inputs, feat_T])


def conv_bn(x, filters, hp_name=None, hp=None):
    hpo_enabled = config['hpo']['enabled']
    if hpo_enabled and hp:
        filters = hp.Int(f'units_{hp_name}', min_value=filters, max_value=filters * 2, step=32)

    activation = hp.Choice(f'activation_{hp_name}', values=config['hpo']['activation']) if hpo_enabled and hp else 'relu'

    x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
    x = layers.BatchNormalization(momentum=0.0)(x)

    return layers.Activation(activation)(x)


def dense_bn(x, filters, hp_name=None, hp=None):
    hpo_enabled = config['hpo']['enabled']
    if hpo_enabled and hp:
        filters = hp.Int(f'units_{hp_name}', min_value=filters, max_value=filters * 2, step=32)

    activation = hp.Choice(f'activation_{hp_name}', values=config['hpo']['activation']) if hpo_enabled and hp else 'relu'

    x = layers.Dense(filters)(x)
    x = layers.BatchNormalization(momentum=0.0)(x)

    return layers.Activation(activation)(x)
