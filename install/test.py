import tensorflow as tf

# Define a simple TensorFlow computation graph
a = tf.constant(1.0, shape=[1000, 1000])
b = tf.constant(2.0, shape=[1000, 1000])
c = tf.matmul(a, b)

# Run the computation graph on the GPU if available
if tf.test.is_gpu_available():
    with tf.device('/gpu:0'):
        result = c.numpy()
else:
    result = c.numpy()

# Print the result and GPU status
print(result)
print(f"Is GPU available? {tf.test.is_gpu_available()}")