import tensorflow as tf
import numpy as np

if __name__=="__main__":
    N_INSTANCE = 10
    MAX_LENGTH = 15
    x = np.random.randint(0, 256, size=(N_INSTANCE, MAX_LENGTH))
    lstm = tf.keras.models.Sequential()
    hidden_state = 16
    lstm.add(
        tf.keras.layers.LSTM(units=hidden_state,
                             return_sequences=True, # True  -> Output Shape is (None, MAX_LENGTH, hidden_state)
                                                    # False -> Output Shape is (None, hidden_state)
                             input_shape=(MAX_LENGTH, 1))
    )
    lstm.summary()