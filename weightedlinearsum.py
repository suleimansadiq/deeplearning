# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import os

# Set the data type and model path
data_t = 'posit8'
model_path = 'posit8.ckpt'

# Define the data type for TensorFlow
if data_t == 'posit32':
    tf_type = tf.posit32
elif data_t == 'posit16':
    tf_type = tf.posit16
elif data_t == 'posit8':
    tf_type = tf.posit8
elif data_t == 'float32':
    tf_type = tf.float32

# Define the LeNet neural network architecture
def LeNet(x):
    # Layer 1: Convolutional Layer
    conv1 = tf.layers.conv2d(x, 20, [5, 5], activation=tf.nn.relu, name='conv1')
    
    # Layer 2: Max Pooling Layer
    pool1 = tf.layers.max_pooling2d(conv1, [2, 2], [2, 2], name='pool1')
    
    # Layer 3: Convolutional Layer
    conv2 = tf.layers.conv2d(pool1, 40, [5, 5], activation=tf.nn.relu, name='conv2')
    
    # Layer 4: Max Pooling Layer
    pool2 = tf.layers.max_pooling2d(conv2, [2, 2], [2, 2], name='pool2')
    
    # Layer 5: Flatten Layer
    flat = tf.layers.flatten(pool2, name='flat')
    
    # Layer 6: Dense Layer
    dense1 = tf.layers.dense(flat, 500, activation=tf.nn.relu, name='dense')
    
    # Layer 7: Dropout Layer
    dropout1 = tf.layers.dropout(dense1, rate=0.5, name='dropout')
    
    # Layer 8: Dense Layer
    logits = tf.layers.dense(dropout1, 10, name='logits')
    
    return logits

# Load the pre-trained model
tf.reset_default_graph()
sess = tf.Session()
if os.path.exists(model_path + '.meta'):
    saver = tf.train.import_meta_graph(model_path + '.meta')
    saver.restore(sess, model_path)
else:
    print(f"Model file {model_path} does not exist. Please train the model first.")

# Define the neural network architecture
x = tf.placeholder(tf_type, (None, 32, 32, 1), name='inputs')
logits = LeNet(x)

# Define the layers dictionary
layers = {
    'Conv2D': {'weights': 'conv1/kernel:0', 'biases': 'conv1/bias:0'},
    'MaxPool': {'weights': None, 'biases': None},
    'Conv2D_1': {'weights': 'conv2/kernel:0', 'biases': 'conv2/bias:0'},
    'MaxPool_1': {'weights': None, 'biases': None},
    'Flatten': {'weights': None, 'biases': None},
    'MatMul': {'weights': 'dense/kernel:0', 'biases': 'dense/bias:0'},
    'MatMul_1': {'weights': 'dense_1/kernel:0', 'biases': 'dense_1/bias:0'},
    'MatMul_2': {'weights': 'dense_2/kernel:0', 'biases': 'dense_2/bias:0'}
}

# Define the weighted linear sums dictionary
weighted_linear_sums = {}

# Get the default graph
graph = tf.get_default_graph()

# Compute the weighted linear sums for each layer
for layer in layers:
    if layers[layer]['weights'] is not None:
        weights_tensor = tf.get_default_graph().get_tensor_by_name(layers[layer]['weights'])
        biases_tensor = tf.get_default_graph().get_tensor_by_name(layers[layer]['biases'])
        print(biases_tensor.shape)
        biases_tensor = tf.reshape(biases_tensor, [1, 1, 1, 20])
        layer_output = graph.get_tensor_by_name(layer + ':0')
        weighted_output = tf.keras.layers.Conv2D(
            filters=20,
            kernel_size=[5, 5],
            strides=[1, 1],
            padding="same",
            name="conv2d"
        )(layer_output)
        weighted_linear_sums[layer] = sess.run(weighted_output, feed_dict={x: np.random.rand(1, 32, 32, 1).astype(np.posit8)})

# Print the weighted linear sums for each layer
for layer, value in weighted_linear_sums.items():
    print(f"Weighted Linear Sum for {layer}:")
    print(value)
