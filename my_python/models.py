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

# %load_ext tensorboard

import shutil



"""## Build the model

"""

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
        self.latent_dim = min(latent_dim, input_dim)   
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(input_dim,)),
            # layers.Dense(20,  kernel_initializer='glorot_normal', activation = 'linear'),
            layers.Dense(latent_dim,  kernel_initializer='glorot_normal', activation = 'linear'),
        ])
        self.decoder = tf.keras.Sequential([
            # layers.Dense(20,  kernel_initializer='glorot_normal', activation = 'linear'),
            layers.Dense(input_dim,  kernel_initializer='glorot_normal', activation = 'linear'),
            # layers.Dense(1,  activation = 'softplus'),
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def get_config(self):
        config = super(Autoencoder, self).get_config()
        config.update({"latent_dim": self.latent_dim})
        return config

"""### Model A: Twin net with custom backpropagation layer"""

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



"""### Model B: Naked Autodiff"""

def get_model_autodiff(input_dim):

    # feedforward
    input_1 = layers.Input(shape=(input_dim,))
    x = layers.Dense(20, kernel_initializer='glorot_normal', activation='softplus', name='FWD_L1')(input_1)
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



"""### Custom autodiff and AE with latent dim 8"""

def get_model_autodiff_AE8(input_dim):

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

    model = tf.keras.models.Model(inputs=input_1, outputs=[y_pred, dydx_pred], name='Autodiff_AE8')
                                          

    return model


"""### Custom autodiff and AE with latent dimension one"""

def get_model_autodiff_AE1(input_dim):

    # feedforward

    latent_dim = 1
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

    model = tf.keras.models.Model(inputs=input_1, outputs=[y_pred, dydx_pred], name='Autodiff_AE1')
                                          

    return model


"""### Learning rate schedules

The original warm-up schedule interpolates the learning rate on a pre-defined grid. The main feature is a steep warm-up in the learning rate at the first epochs.

Subclasses of LearningRateSchedule require that code can be run in graph mode (see `@tf.function` decorator). This is not the case for the python interpolation method used in the original warm-up schedule. One alternative is to implement the learning rate schedule via a callback. For consistency with other schedulers this implementation performs the interpolation upfront at initialisation.
"""

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
        scaled_MSE,
        differential_weight=1,
        lr_schedule = lr_warmup,
        alpha = None
    ):

    model = model_getter(input_dim)
    if alpha is not None:
        alpha = 1.0 / (1.0 + differential_weight * input_dim)
    else:
        alpha = alpha


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

    return model

"""## Training

### Data normalization: pre- and post-processing

Description of Differential PCA in Huge/Savine [Appendix 2](https://https://github.com/differential-machine-learning/appendices/blob/master/App2-Preprocessing.pdf)

Code chunks of PCA courteously provided by Antoine Savine.
"""

class DPCALayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(DPCALayer, self).__init__(**kwargs)

        self.x_eig_thresh  = 1.0e-04   # filter threshold for const/redundant inputs
        self.dx_eig_thresh = 2.0e-02   # filter threshold for zero/redundant derivatives

        self.muX = 0
        self.muY = 0 
        self.stdY = 0
        self.dydxTrainScaled = 0
        self.xTrainScaled = 0
        self.x1Tox3 = 0
        self.dydxTrainScaled = 0
        self.x1BarTox3Bar = 0 
        self.x3BarTox1Bar = 0
        self.n = 0

    
    def adapt(self, x_raw, y_raw, dydx_raw):
        # basic processing (step 1 in the note)
        
        x0 = x_raw
        y0 = y_raw
        x0Bar = dydx_raw
        
        n0 = x_raw.shape[1]  
        m = x_raw.shape[0] # size of training set

        # center inputs
        # compute
        self.muX = x0.mean(axis=0)
        # apply
        x1 = x0 - self.muX
               
        # normalize inputs
        # compute
        self.muY = y0.mean(axis=0)
        self.stdY = y0.std(axis=0)
        # apply
        y1 = (y0 - self.muY) / self.stdY
       
        # update derivs
        x1Bar = x0Bar / self.stdY
        
        # dim
        n1 = n0
                
        # input orthonormalization and filtering (step 2 in the note)
        
        # eigenvalue decomposition
        d2, p2 = np.linalg.eigh(x1.T @ x1 / m)
        
        # filter
        f2 = np.argwhere(d2 > self.x_eig_thresh**2).reshape(-1)

        # dim       
        n2 = f2.size
        if n2 == 0:
            raise Exception("all variables were filtered out in step 2")
        
        # shrink eigenvalues and eigenvectors by filter
        d2Tilde = d2[f2]
        p2Tilde = p2[:, f2]
        
        # scale inputs
        # compute
        sqrtD2Tilde = np.sqrt(d2Tilde).reshape((1,-1))
        x1Tox2 = p2Tilde / sqrtD2Tilde
        # apply 
        x2 = x1 @ x1Tox2 
        
        # update derivs
        # compute
        x1BarTox2Bar = p2Tilde * sqrtD2Tilde
        x2BarTox1Bar = p2Tilde.T / sqrtD2Tilde.reshape((-1,1))
        # apply
        x2Bar = x1Bar @ x1BarTox2Bar               
        # derivatives orthogonalization and filtering (step 3 in the note)
        
        # eigenvalue decomposition
        d3, p3 = np.linalg.eigh(x2Bar.T @ x2Bar / m)
            
        # filter
        f3 = np.argwhere(d3 > self.dx_eig_thresh**2).reshape(-1)

        # dim       
        n3 = f3.size
        if n3 == 0:
            raise Exception("all variables were filtered out in step 3")
            
        # shrink eigenvectors by filter
        p3Tilde = p3[:, f3]
            
        # scale inputs
        # compute
        x2Tox3 = p3Tilde
        # apply
        x3 = x2 @ x2Tox3
        
        # update derivs
        # compute
        x2BarTox3Bar = p3Tilde
        x3BarTox2Bar = p3Tilde.T
        # apply
        x3Bar = x2Bar @ x2BarTox3Bar

        # raw to processed and back (step 4 in the note)
    
        # y
        self.yTrainScaled = y1
        
        # x
        self.xTrainScaled = x3
        # scaling matrix
        self.x1Tox3 = x1Tox2 @ x2Tox3
        
        # dx
        self.dydxTrainScaled = x3Bar
        # scaling
        self.x1BarTox3Bar = x1BarTox2Bar @ x2BarTox3Bar
        # and back
        self.x3BarTox1Bar = x3BarTox2Bar @ x2BarTox1Bar

        # dim
        self.n = n3

    def call(self, inputs):
        # layer called on x as inputs
        return (inputs - self.muX) @ self.x1Tox3
    
    def yScaled(self, y):
        return (y - self.muY) / self.stdY

    def yScaledInverse(self, y):
        return (y * self.stdY )  + self.muY

    def dydxScaled(self, dydx):
        return (dydx / self.stdY) @ self.x1BarTox3Bar
    
    def dydxScaledInverse(self, dydx_scaled):
        return dydx_scaled @ self.x3BarTox1Bar * self.stdY

    def output_n(self):
        return self.n

    def get_config(self):
        config = super(DPCALayer, self).get_config()
        return config

class NormalisationLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(NormalisationLayer, self).__init__(**kwargs)

        self.muX = 0
        self.muY = 0 
        self.stdX = 0
        self.stdY = 0

        self.n = 0

    def adapt(self, x_raw, y_raw, dydx_raw):
        # basic processing (step 1 in the note)
        
        x0 = x_raw
        y0 = y_raw
        x0Bar = dydx_raw
        
        self.n = x_raw.shape[1] 
        m = x_raw.shape[0] # size of training set

        # normalize inputs
        # compute
        self.muX = x0.mean(axis=0)
        self.stdX = x0.std(axis=0)
       
        # normalize inputs
        # compute
        self.muY = y0.mean(axis=0)
        self.stdY = y0.std(axis=0)
        

    def call(self, inputs):
        # layer called on x as inputs
        return (inputs - self.muX) /  self.stdX
    
    def yScaled(self, y):
        return (y - self.muY) / self.stdY

    def yScaledInverse(self, y):
        return (y * self.stdY ) + self.muY

    def dydxScaled(self, dydx):
        return dydx * (self.stdX / self.stdY)
    
    def dydxScaledInverse(self, dydx_scaled):
        return dydx_scaled * (self.stdY / self.stdX)
    
    def output_n(self):
        return self.n

    def get_config(self):
        config = super(NormalisationLayer, self).get_config()
        return config

class NoNormalisationLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(NoNormalisationLayer, self).__init__(**kwargs)

        self.n = 0

    def adapt(self, x_raw, y_raw, dydx_raw):
        # basic processing (step 1 in the note)
        
       
        self.n = x_raw.shape[1] 
        

    def call(self, inputs):
        # layer called on x as inputs
        return inputs

    def yScaled(self, y):
        return y

    def yScaledInverse(self, y):
        return y

    def dydxScaled(self, dydx):
        return dydx 

    def dydxScaledInverse(self, dydx_scaled):
        return dydx_scaled 

    def output_n(self):
        return self.n

    def get_config(self):
        config = super(NoNormalisationLayer, self).get_config()
        return config

def preprocess_data(x_train, y_train, dydx_train, prep_type='Normalisation'):

    if (prep_type == 'PCA'):
        prep_layer = DPCALayer(input_shape=[x_train.shape[1],])
    elif (prep_type == 'Normalisation'):
        prep_layer = NormalisationLayer(input_shape=[x_train.shape[1],])
    elif (prep_type == 'NoNormalisation'):
        prep_layer = NoNormalisationLayer(input_shape=[x_train.shape[1],])
    else:
        print('Pre-processing unknwon. Use no normalisation instead')
        prep_layer = NormalisationLayer(input_shape=[x_train.shape[1],])
    
    prep_layer.adapt(x_train, y_train, dydx_train)

    scaled_MSE = L2ScaledMSE()
    scaled_MSE.adapt(prep_layer(dydx_train))
 
    return prep_layer, scaled_MSE

# predict and inverse transform   
def predict_unscaled(model, prep_layer, x_unscaled):

    y_scaled, dydx_scaled = model.predict(prep_layer(x_unscaled))
    y_pred = prep_layer.yScaledInverse(y_scaled)
    dydx_pred = prep_layer.dydxScaledInverse(dydx_scaled)
    
    return y_pred.reshape(-1,1), dydx_pred

"""### Training utility



"""



BATCH_SIZE = 1024
EPOCHS = 200
def train_model(model,
                prep_layer, 
                train_id,
                x_train, 
                y_train, 
                dydx_train=None,
                epochs = EPOCHS,
                batch_size = BATCH_SIZE,
                x_true = None,
                y_true = None,
                dydx_true = None):

    
    history = model.fit(
        prep_layer(x_train), [prep_layer.yScaled(y_train), prep_layer.dydxScaled(dydx_train)], 
        # steps_per_epoch = STEPS_PER_EPOCH,
        batch_size = batch_size,
        epochs=epochs,
        callbacks=[
                   tf.keras.callbacks.TensorBoard(log_dir = log_dir+train_id, histogram_freq=1),
                   tf.keras.callbacks.EarlyStopping(monitor='loss',patience=100),
                   TqdmCallback(verbose=1)
                   ],
        validation_data = (prep_layer(x_true), [prep_layer.yScaled(y_true), prep_layer.dydxScaled(dydx_true)]),
        verbose=0
        )
    return history

"""## Examples of Twin Net and Autodiff AE

### Impact of sample size
"""
