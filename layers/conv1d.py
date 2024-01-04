import numpy as np
import tensorflow as tf

if __name__=="__main__":
    N_INSTANCE = 100
    N_FEATURES = 42
    N_FEATURE_SIZE = 15
    
    conv1d = tf.keras.models.Sequential()
    filter = 100
    kernel_size = 3
    activation = 'relu'
    conv1d.add(
        tf.keras.layers.Conv1D(
            filters=filter,
            kernel_size=kernel_size,
            activation=activation,
            input_shape = (N_FEATURES, N_FEATURE_SIZE) # critical
        )
    )
    # _________________________________________________________________
    #  Layer (type)                Output Shape              Param #   
    # =================================================================
    #  conv1d (Conv1D)             (None, 40, 32)            4600      

    # =================================================================
    # Total params: 4600 (17.97 KB)
    # Trainable params: 4600 (17.97 KB)
    # Non-trainable params: 0 (0.00 Byte)
    # _________________________________________________________________
    conv1d.summary()