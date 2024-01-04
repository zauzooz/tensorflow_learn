import tensorflow as tf

import tensorflow as tf

class CustomGradientTape(tf.GradientTape):
    def __init__(self, persistent=False, watch_accessed_variables=True):
        super(CustomGradientTape, self).__init__(persistent=persistent, watch_accessed_variables=watch_accessed_variables)

    def gradient_with_hook(self, target, sources, output_gradients=None):
        """
        Compute the gradient of target with respect to sources, applying a custom hook.
        """
        gradients = super(CustomGradientTape, self).gradient(target, sources, output_gradients)

        # Apply your custom hook to modify gradients
        modified_gradients = [self.explanation_hook(g, v) for g, v in zip(gradients, sources)]

        return tf.convert_to_tensor(modified_gradients)

    def explanation_hook(self, gradient, variable):
        # Implement your custom explanation hook logic here
        # For example, you can modify the gradients or perform other operations
        modified_gradient = gradient * 0.8  # Modify the gradient (this is just an example)

        return modified_gradient

if __name__ == "__main__":
    # Example usage:
    x = tf.constant([3.0, 1.2])
    y = x**2

    with tf.GradientTape() as tape:
        # Watch the variable x
        tape.watch(x)

        # Define a computation using x
        y = x**2

        # Compute the gradient of y with respect to x
        original_gradient = tape.gradient(y, x)

    del tape

    # Create a custom gradient tape
    with CustomGradientTape() as tape:
        # Watch the variable x
        tape.watch(x)

        # Define a computation using x
        y = x**2

        # Compute the gradient of y with respect to x and apply the custom hook
        modified_gradient = tape.gradient_with_hook(y, x)
    del tape
    print("Original Gradient:", original_gradient.numpy())
    print("Modified Gradient:", modified_gradient.numpy())