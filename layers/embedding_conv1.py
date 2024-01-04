import tensorflow as tf
import numpy as np
if __name__ =="__main__":
    N_SAMPLE = 100
    N_FEATURES = 50
    x = np.random.randint(0,255,size=(N_SAMPLE, N_FEATURES))
    model = tf.keras.Sequential()
    vocab_size = 256
    output_dim = 64
    model.add(tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=output_dim, input_length=N_FEATURES))
    model.add(tf.keras.layers.Conv1D( filters=32, kernel_size=3, activation='relu', input_shape=(N_FEATURES, output_dim)))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=3))
    model.add(tf.keras.layers.Conv1D( filters=32, kernel_size=3, activation='relu', input_shape=(N_FEATURES, output_dim)))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=3))
    model.add(tf.keras.layers.Flatten())
    model.compile('rmsprop', 'mse')
    x1 = model.predict(x)
    model.save('model.h5')