import numpy as np
import tensorflow as tf

if __name__ == "__main__":
    model = tf.keras.models.Sequential()
    N_INSTANCES = 3,
    N_FEATURES = 10
    # x = np.random.randint(0,256,size=(N_INSTANCES, N_FEATURES))
    vocab_size = 256
    output_dim = 128
    model.add(
        tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=output_dim,
            input_length=N_FEATURES
        )
    )
    # Layer (type)                Output Shape              Param #   
    # =================================================================
    #  embedding (Embedding)       (None, 10, 128)           32768     

    # =================================================================
    # Total params: 32768 (128.00 KB)
    # Trainable params: 32768 (128.00 KB)
    # Non-trainable params: 0 (0.00 Byte)
    # _________________________________________________________________
    model.compile(
        'rmsprop', 'mse'
    )
    model.summary()