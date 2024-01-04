import tensorflow as tf

class MultiHeadAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads, d_model, threshold, **kwargs):
        super(MultiHeadAttentionLayer, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        self.threshold = threshold
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.depth = d_model // num_heads

    def build(self, input_shape):
        super(MultiHeadAttentionLayer, self).build(input_shape)

        # Create linear layers for each head
        self.wq = [tf.keras.layers.Dense(self.depth) for _ in range(self.num_heads)]
        self.wk = [tf.keras.layers.Dense(self.depth) for _ in range(self.num_heads)]
        self.wv = [tf.keras.layers.Dense(self.depth) for _ in range(self.num_heads)]

        # Linear layer to project the concatenated outputs
        self.dense = tf.keras.layers.Dense(self.d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # Inputs should be a tuple containing T and Q
        Q, T, V = inputs

        batch_size = tf.shape(Q)[0]

        # Split Q, K, and V into multiple heads
        q = [w(Q) for w in self.wq]
        k = [w(T) for w in self.wk]
        v = [w(V) for w in self.wv]

        # Split the heads
        # q = [self.split_heads(head, batch_size) for head in q]
        # k = [self.split_heads(head, batch_size) for head in k]
        # v = [self.split_heads(head, batch_size) for head in v]

        # Compute the attention scores for each head
        outputs = []
        for i in range(self.num_heads):
            scaled_attention = self.calculate_attention(q[i], k[i], v[i])
            outputs.append(scaled_attention)

        # Concatenate the outputs of all heads
        concat_attention = tf.concat(outputs, axis=-1)

        # Project the concatenated outputs
        output = self.dense(concat_attention)

        return output

    def calculate_attention(self, q, k, v):
        # Compute the attention score matrix
        A = tf.matmul(q, k, transpose_b=True)  # Q * K^T

        threshold_mask = tf.cast(A > self.threshold, A.dtype)
        A = tf.where(A > self.threshold, A, -2**32 * threshold_mask)

        sequence_length = tf.shape(k)[-2]

        # Create the mask matrix M based on the shape of Q and K
        M = 1 - tf.linalg.band_part(tf.ones((sequence_length, sequence_length)), -1, 0)
        M = tf.expand_dims(M, axis=0)  # Add batch dimension to the mask

        A = A + M

        d_k = tf.cast(tf.shape(k)[-1], A.dtype)  # Get the dimension d_k from the last dimension of Q (K)

        # Apply the scaling factor
        A = A / tf.sqrt(d_k)

        # Apply softmax to the scaled attention scores
        y = tf.keras.activations.softmax(A, axis=-1)

        # Multiply with V to get the final output
        output = tf.matmul(y, v)

        return output

# Example usage:
m = 15
sequence_length = 255
feature_dim = 96

# Create random T and Q tensors for testing
T = tf.constant(tf.random.normal((m, sequence_length, feature_dim)))
V = tf.constant(tf.random.normal((m, sequence_length, feature_dim)))
Q = tf.constant(tf.random.normal((m, sequence_length, feature_dim)))

# Create the CustomAttentionLayer
attention_layer = MultiHeadAttentionLayer(num_heads=16, d_model=feature_dim, threshold=0.1)

# Compute the attention score matrix A
A = attention_layer([T, Q, V])

print("Input T Shape:", T.shape)
print("Input Q Shape:", Q.shape)
print("Input Q Shape:", V.shape)
print("Output A Shape:", A.shape)