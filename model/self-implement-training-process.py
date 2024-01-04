import tensorflow as tf
import numpy as np

# self implement model1 template

class SelfModel:
    def __init__(self, input_dim) -> None:
        self.model1 = tf.keras.models.Sequential()
        self.model1.add(tf.keras.layers.Dense(units=10, activation='relu', input_shape=(input_dim,)))
        self.model1.add(tf.keras.layers.Dense(units=2, activation='softmax'))
        self.loss = tf.keras.losses.CategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.SGD()
    
    def compile(self):
        self.model1.compile(optimizer=self.optimizer)
    
    def fit(self, x: np.ndarray, y: np.ndarray, nb_epochs, batch_size):
        def update_lr():
            pass

        def batch_set(X1: np.ndarray, y: np.ndarray, batch_size):
            X1_batchs = []
            y_batchs = []
            if batch_size is None:
                X1_batchs.append(X1)
                y_batchs.append(y)

            else: 
                N = X1.shape[0]
                indexes = list(range(N))
                np.random.shuffle(indexes)
                X1_batchs = [X1[indexes[i:i+batch_size]] for i in range(0, N, batch_size)]
                y_batchs = [y[indexes[i:i+batch_size]] for i in range(0, N, batch_size)]
            return X1_batchs, y_batchs

        def gradient_descent_nn(model1, batch_inputs, batch_targets):
            epoch_loss = tf.keras.metrics.Mean()
            with tf.GradientTape() as tape:
                predictions = model1(batch_inputs, training=True)
                loss = self.loss(batch_targets, predictions)
                gradients = tape.gradient(loss, model1.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, model1.trainable_variables))
            epoch_loss(loss)
            return (epoch_loss.result().numpy(), model1)
        
        for epoch in range(nb_epochs):
            x_batchs, y_batchs = batch_set(X1=x, y=y, batch_size=batch_size)
            for x_batch, y_batch in zip(x_batchs, y_batchs):
                loss, self.model1 = gradient_descent_nn(model1=self.model1, batch_inputs=x_batch, batch_targets=y_batch)
    
    def predict(self, x:np.ndarray):
        pass

    def evaluate(self, x:np.ndarray, y:np.ndarray):
        pass
    
    def summary(self, x:np.ndarray):
        pass

    def save(self, model_path):
        pass

    def load(self, model_path):
        pass