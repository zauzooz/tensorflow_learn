import tensorflow as tf

class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, threshold, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.threshold = threshold  

    def build(self, input_shape):
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # Inputs should be a tuple containing T and Q
        Q, T, V = inputs

        # Compute the attention score matrix
        K = tf.transpose(T, perm=[0, 2, 1])  # Transpose T to match dimensions with Q
        A = tf.matmul(Q, K)  # Q * K^T

        threshold_mask = tf.cast(A > self.threshold, A.dtype)
        A = tf.where(A > self.threshold, A, -2**32 * threshold_mask)

        m = Q.shape[0]
        sequence_length = Q.shape[1]
        M = self.create_mask_matrix(m, sequence_length) * -2**32

        A = A + M

        d_k = tf.cast(Q.shape[-1], A.dtype)  # Get the dimension d_k from the last dimension of Q (K)

        # Apply the scaling factor
        A = A / tf.sqrt(d_k)

        # Apply softmax to the scaled attention scores
        y = tf.keras.activations.softmax(A, axis=-1)

        # Multiply with V to get the final output
        output = tf.matmul(y, V)

        return output
    
    def create_mask_matrix(self, m, sequence_length):
        # Create the mask matrix M based on the shape of Q and K
        # Here's an example that generates the mask matrix you described:
        M = 1 - tf.linalg.band_part(tf.ones((m, sequence_length, sequence_length)), -1, 0)
        return M

# Example usage:
m = 15
sequence_length = 255
feature_dim = 96

# Create random T and Q tensors for testing
T = tf.constant(tf.random.normal((m, sequence_length, feature_dim)))
V = tf.constant(tf.random.normal((m, sequence_length, feature_dim)))
Q = tf.constant(tf.random.normal((m, sequence_length, feature_dim)))

# Create the CustomAttentionLayer
attention_layer = AttentionLayer(threshold=0.3)

# Compute the attention score matrix A
A = attention_layer([T, Q, V])

print("Input T Shape:", T.shape)
print("Input Q Shape:", Q.shape)
print("Input Q Shape:", V.shape)
print("Output A Shape:", A.shape)
