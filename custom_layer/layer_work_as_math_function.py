import tensorflow as tf

class Gnar(tf.keras.layers.Layer):
    def __init__(self):
        super(Gnar, self).__init__()

    def call(self, inputs):
        # Get the batch size, sequence length, and feature dimension
        batch_size, sequence_length, feature_dim = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2]

        # Create a range of indices from 1 to sequence_length
        indices = tf.range(1, sequence_length + 1, dtype=tf.float32)

        # Expand the indices to match the batch size and feature dimension
        indices = tf.tile(tf.expand_dims(indices, 0), [batch_size, 1])
        indices = tf.expand_dims(indices, -1)

        # Calculate the cumulative sum along the sequence dimension
        cumulative_sum = tf.cumsum(inputs, axis=1)

        # Divide the cumulative sum by the indices to get the desired output
        output = cumulative_sum / indices

        return output

# Example usage:
input_data = tf.constant(tf.random.normal((10, 255, 96)))  # Replace with your input data
gnar_layer = Gnar()
output = gnar_layer(input_data)

print("Input Shape:", input_data.shape)
print("Output Shape (Gnar):", output.shape)