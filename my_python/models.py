# -*- coding: utf-8 -*-


# Commented out IPython magic to ensure Python compatibility.
# import and test
import tensorflow as tf
print('TF version: ',tf.__version__) # works with 2.4

import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm.keras import TqdmCallback

from tensorflow import keras
from tensorflow.keras import layers

import datetime             

import pathlib
import shutil

tf.keras.backend.set_floatx('float32') # default
real_type = tf.float32


"""## Build the model

### Model A: Twin net with custom backpropagation layer
"""

class BackpropDense(tf.keras.layers.Layer):
    def __init__(self, reference_layer, **kwargs):
      super(BackpropDense, self).__init__(**kwargs)
      self.units = None
      self.ref_layer = reference_layer # weights of ref layer 'collected' by tensorflow
    
    def call(self, gradient, z):
        if z is not None:
            # essential backprop equation
            gradient = tf.matmul(gradient, tf.transpose(self.weights[0])) * tf.math.sigmoid(z)
        else:
            gradient = tf.matmul(gradient, tf.transpose(self.weights[0]))
        return gradient
    
    def get_config(self):
        config = super(BackpropDense, self).get_config()
        config.update({"reference_layer": self.ref_layer})
        return config


def get_model_twin_net(input_dim):

    # feedforward

    # init fwd layers explicitely, convenient for reference in backprop
    layer_1 = layers.Dense(20, kernel_initializer='glorot_normal', activation = None, name='FWD_L1')
    layer_2 = layers.Dense(20, kernel_initializer='glorot_normal', activation = None, name='FWD_L2')
    layer_3 = layers.Dense(20, kernel_initializer='glorot_normal', activation = None, name='FWD_L3')
    layer_4 = layers.Dense(20, kernel_initializer='glorot_normal', activation = None, name='FWD_L4')
    layer_5 = layers.Dense(1, kernel_initializer='glorot_normal', activation = 'linear', name='y_pred')

    # apply activation function on input to preceeding layer, we need non-activated output for backprop
    input_1 = layers.Input(shape=(input_dim,))
    x1 = layer_1(input_1)
    x2 = layer_2(layers.Activation('softplus', name="Act_1")(x1))
    x3 = layer_3(layers.Activation('softplus', name="Act_2")(x2))
    x4 = layer_4(layers.Activation('softplus', name="Act_3")(x3))
    y_pred = layer_5(layers.Activation('softplus', name="Act_4")(x4))

    # backprop

    # backprop has no trainable weights, only references to layers in forward net
    grad = BackpropDense(layer_5,name='Bck_L1')(tf.ones_like(y_pred), x4)
    grad = BackpropDense(layer_4,name='Bck_L2')(grad, x3)
    grad = BackpropDense(layer_3,name='Bck_L3')(grad, x2)
    grad = BackpropDense(layer_2,name='Bck_L4')(grad, x1)
    dydx_pred = BackpropDense(layer_1,name='dydx_pred')(grad, None)

    model = tf.keras.models.Model(inputs=input_1, outputs=[y_pred, dydx_pred], name='Twin_Net')

    return model



"""### Model B: Custom autodiff and autoencoder"""

from tensorflow.keras import regularizers

class AutodiffLayer(tf.keras.layers.Layer):
    def __init__(self, fwd_model, **kwargs):
      super(AutodiffLayer, self).__init__(**kwargs)
      self.units = None
      self.fwd_model = fwd_model # weights of ref layer 'collected' by tensorflow
    
    def call(self, input):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(input)
            pred_value = self.fwd_model(input)
           
        # Get the gradients of the loss w.r.t to the pricing inputs
        gradient = tape.gradient(pred_value, input)

        return gradient
    
    def get_config(self):
        config = super(AutodiffLayer, self).get_config()
        config.update({"fwd_model": self.fwd_model})
        return config

class Autoencoder(tf.keras.layers.Layer):
    def __init__(self, input_dim, latent_dim, **kwargs):
        super(Autoencoder, self).__init__(**kwargs)
        self.latent_dim = latent_dim   
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(20,  kernel_initializer='glorot_normal', activation = 'linear'),
            layers.Dense(latent_dim,  kernel_initializer='glorot_normal', activation = 'softplus'),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(20,  kernel_initializer='glorot_normal', activation = 'linear'),
            layers.Dense(1,  activation = 'softplus'),
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def get_config(self):
        config = super(Autoencoder, self).get_config()
        config.update({"latent_dim": self.latent_dim})
        return config


def get_model_autodiff_autoencoder(input_dim):

    # feedforward

    latent_dim = 8
    auto_encoder = Autoencoder(input_dim, latent_dim, name='auto_encoder')

    input_1 = layers.Input(shape=(input_dim,))
    x = auto_encoder(input_1)
    x = layers.Dense(20, kernel_initializer='glorot_normal', activation='softplus', name='FWD_L1')(x)
    x = layers.Dense(20, kernel_initializer='glorot_normal', activation='softplus', name='FWD_L2')(x)
    x = layers.Dense(20, kernel_initializer='glorot_normal', activation='softplus', name='FWD_L3')(x)
    x = layers.Dense(20, kernel_initializer='glorot_normal', activation='softplus', name='FWD_L4')(x)
    y_pred = layers.Dense(1, kernel_initializer='glorot_normal', activation='linear', name='y_pred')(x)

    fwd_model = tf.keras.models.Model(inputs=input_1, outputs=y_pred)

    # autodiff

    autodiff_layer = AutodiffLayer(fwd_model, name='dydx_pred')
    dydx_pred = autodiff_layer(input_1)

    model = tf.keras.models.Model(inputs=input_1, outputs=[y_pred, dydx_pred], name='Autodiff_Autoencoder')
                                          

    return model


"""### Learning rate schedules

The original warm-up schedule interpolates the learning rate on a pre-defined grid. The main feature is a steep warm-up in the learning rate at the first epochs.

Subclasses of LearningRateSchedule require that code can be run in graph mode (see `@tf.function` decorator). This is not the case for the python interpolation method used in the original warm-up schedule. One alternative is to implement the learning rate schedule via a callback. For consistency with other schedulers this implementation performs the interpolation upfront at initialisation.
"""

EPOCHS = 100
STEPS_PER_EPOCH = 16

class WarmUpSchedule(tf.optimizers.schedules.LearningRateSchedule):
    def __init__(self, steps_per_epoch):
        super(WarmUpSchedule, self).__init__()
        self.ms = 100 * steps_per_epoch # original schedule calibrated to 100 epochs
        boundaries = (0.0, 0.2, 0.6, 0.9, 1.0)
        values = (1e-08, 0.1, 0.01, 1e-06, 1e-08)
        self.interp = tf.convert_to_tensor(
            [np.interp(step / self.ms, boundaries, values) for step in np.arange(0,self.ms)],
            dtype=tf.float32
            )

    @tf.function
    def __call__(self, step):
        if (tf.cast(step, tf.int32) >= self.ms):
            return 1e-08
        else:
            return tf.gather(self.interp, tf.cast(step, tf.int32))

lr_warmup = WarmUpSchedule(STEPS_PER_EPOCH)

"""A standard *inverse time decay* schedule is provided as an alternative. The initial learning rate is calibrated on the basket option discussed later."""

lr_inv_time_decay = tf.keras.optimizers.schedules.InverseTimeDecay(
  0.01,
  decay_steps=STEPS_PER_EPOCH*100,
  decay_rate=50,
  staircase=False)


"""### Custom loss function

Custom implementation of the MSE loss function for the differential labels. Follows the original approach to weight losses by L2 norm of differentials to level out different scales. The scale of the differentials is solely determined by the functional dependency to the price parameters.
"""

class L2ScaledMSE(keras.losses.Loss):
# normalize ith component loss by L2 norm to level MSE contribution
    def __init__(self, norm_weights = None, name="L2ScaleddMSE"):
        super().__init__(name=name)
        self.norm_weights = norm_weights

    def adapt(self, dydx_train):
        arg = tf.convert_to_tensor(dydx_train, dtype=tf.float32)
        self.norm_weights = 1.0 / tf.reshape(tf.sqrt(tf.reduce_mean(arg ** 2 ,axis=0)),[1,-1])

    @tf.function
    def call(self, y_true, y_pred):
        return tf.math.reduce_mean(tf.square((y_true  - y_pred) * self.norm_weights))

"""### Compile model"""

def build_and_compile_model(
        input_dim,
        model_getter,
        alpha=0.5,
        lr_schedule = lr_warmup
    ):

    model = model_getter(input_dim)
    scaled_MSE = L2ScaledMSE()

    # build model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr_schedule),
        loss={ # named losses
                'y_pred': 'mse',
                'dydx_pred' : scaled_MSE
            },
        run_eagerly=None,
        loss_weights=[alpha,1-alpha]
    )

    return model, scaled_MSE

"""## Training

### Data normalization: pre- and post-processing
"""

# initialize and calibrate data normalisation layers 
def get_norm_layers(x_raw, y_raw, dydx_raw=None):

    # initalize normalisation layer
    
    x_dim = x_raw.shape[1]
    y_dim = y_raw.shape[1]
    x_norm = tf.keras.layers.experimental.preprocessing.Normalization(input_shape=[x_dim,])
    y_norm = tf.keras.layers.experimental.preprocessing.Normalization(input_shape=[y_dim,])
    if dydx_raw is not None:
        dydx_norm = tf.keras.layers.experimental.preprocessing.Normalization(input_shape=[x_dim,])
    else:
        dydx_norm = None
   
    # calibrate
    
    x_norm.adapt(x_raw)
    y_norm.adapt(y_raw)
    if dydx_norm is not None:
        _ = dydx_norm(dydx_raw) # init normalizer with mean 0 and variance 1
        dydx_norm.set_weights([np.zeros(x_dim), y_norm.variance.numpy() / x_norm.variance.numpy()])
    
    return x_norm, y_norm, dydx_norm

# predict and inverse transform   
def predict_unscaled(model, x_norm, y_norm, x_unscaled):

    y_scaled, dydx_scaled = model.predict(x_norm(x_unscaled))
    y_pred = y_norm.mean + tf.sqrt(y_norm.variance) * y_scaled
    dydx_pred = tf.sqrt(tf.math.divide_no_nan(y_norm.variance, x_norm.variance)) * dydx_scaled
    
    return y_pred, dydx_pred

"""### Training utility



"""

def train_model(model,
                train_id, 
                x_train, 
                y_train, 
                dydx_train=None,
                scaled_MSE=None, 
                epochs = EPOCHS,
                x_true = None,
                y_true = None,
                dydx_true = None):
   
    x_norm, y_norm, dydx_norm = get_norm_layers(x_train, y_train, dydx_train)
    if scaled_MSE is not None:
        scaled_MSE.adapt(dydx_train)
    
    history = model.fit(
        x_norm(x_train), [y_norm(y_train), dydx_norm(dydx_train)], 
        steps_per_epoch = STEPS_PER_EPOCH,
        epochs=epochs,
        callbacks=[
                   # tf.keras.callbacks.TensorBoard(log_dir = log_dir+train_id, histogram_freq=1),
                   tf.keras.callbacks.EarlyStopping(monitor='loss',patience=100),
                   TqdmCallback(verbose=1)
                   ],
        validation_data = (x_norm(x_true), [y_norm(y_true), dydx_norm(dydx_true)]),
        verbose=0
        )
    return history, x_norm, y_norm

