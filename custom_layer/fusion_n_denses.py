import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Dense, Input
from tensorflow.keras.constraints import Constraint

# Define a custom constraint for wc and wr
class ConstrainedWeight(Constraint):
    def __init__(self):
        super(ConstrainedWeight, self).__init__()

    def __call__(self, w):
        return tf.clip_by_value(w, 0, 1)

class WeightedElementwiseAddition(Layer):
    def __init__(self, **kwargs):
        super(WeightedElementwiseAddition, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create trainable variables wc and wr with custom constraint
        self.wc = self.add_weight(name='wc', shape=(), initializer='random_normal', trainable=True, constraint=ConstrainedWeight())
        self.wr = self.add_weight(name='wr', shape=(), initializer='random_normal', trainable=True, constraint=ConstrainedWeight())

        super(WeightedElementwiseAddition, self).build(input_shape)

    def call(self, inputs):
        result = self.wc * inputs[0] + self.wr * inputs[1]
        return result

if __name__=="__main__":
    input1 = Input((41))
    input2 = Input((41))
    fusion_layer = WeightedElementwiseAddition()([input1,input2])
    # Add additional dense layers to the model
    dense1 = Dense(128, activation='relu')(fusion_layer)
    dense2 = Dense(64, activation='relu')(dense1)
    output = Dense(3, activation='softmax')(dense2)  # Adjust the number of output units and activation function as needed
    model = tf.keras.Model([input1,input2],output)
    model.summary()
    model.save("fusion_n_denses.h5")
    del model
    tf.keras.utils.get_custom_objects()['WeightedElementwiseAddition'] = WeightedElementwiseAddition
    _model = tf.keras.models.load_model("fusion_n_denses.h5")
    _model.summary()