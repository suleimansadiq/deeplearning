# -*- coding: utf-8 -*-
"""
![LeNet Architecture](lenet.png)
Source: Yan LeCun
"""

from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.contrib.layers import flatten
import sys
import csv
import os
import time

# Function to save activations to CSV
def save_activations(activations, layer_name, batch_index):
    activations_dir = f'./activation_data/{layer_name}'
    if not os.path.exists(activations_dir):
        os.makedirs(activations_dir)
    activations_file = f'{activations_dir}/batch_{batch_index}.csv'
    with open(activations_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(activations)

def save_layer_data(layer_name, data):
    csv_filename = f'./csv_data/{layer_name}.csv'
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(data.keys())
        csv_writer.writerows(zip(*data.values()))

# Remove this for further evaluation
np.random.seed(1)
tf.set_random_seed(2)

# Load Dataset
if len(sys.argv) > 2:
    data_set = sys.argv[2]
    if sys.argv[2] == 'mnist':
        d_set = mnist
    elif sys.argv[2] == 'fashion_mnist':
        d_set = fashion_mnist
else:
    data_set = 'mnist'
    d_set = mnist

# confirm Dataset
print("Dataset is: ", data_set)

(X_train, y_train), (X_test, y_test) = d_set.load_data()
X_train = np.expand_dims(X_train, axis=3)  # (60000, 28, 28, 1)
X_test = np.expand_dims(X_test, axis=3)  # (10000, 28, 28, 1)

assert(len(X_train) == len(y_train))
assert(len(X_test) == len(y_test))

# X_train = X_train[:128]
# y_train = y_train[:128]
# X_test = X_test[:128]
# y_test = y_test[:128]

print("\nImage Shape: {}\n".format(X_train[0].shape))
print("Training Set:   {} samples".format(len(X_train)))
print("Test Set:       {} samples".format(len(X_test)))

"""The MNIST data that TensorFlow pre-loads comes as 28x28x1 images.

However, the LeNet architecture only accepts 32x32xC images, where C is the number of color channels.

In order to reformat the MNIST data into a shape that LeNet will accept, we pad the data with two rows of zeros on the top and bottom, and two columns of zeros on the left and right (28+2+2 = 32).

You do not need to modify this section.
"""

# Pad images with 0s
X_train = np.pad(X_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
X_test = np.pad(X_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

print("Updated Image Shape: {}".format(X_train[0].shape))

"""## Setup TensorFlow
The `EPOCH` and `BATCH_SIZE` values affect the training speed and model accuracy.
"""

EPOCHS = 1
BATCH_SIZE = 128
print('Total epochs:', EPOCHS)

# Set Posit data types
if len(sys.argv) > 1:
    data_t = sys.argv[1]
    if sys.argv[1] == 'posit32':
        eps = 1e-8
        posit = np.posit32
        tf_type = tf.posit32
    elif sys.argv[1] == 'posit16':
        eps = 1e-4
        posit = np.posit16
        tf_type = tf.posit16
    elif sys.argv[1] == 'posit8':
        eps = 0.015625
        posit = np.posit8
        tf_type = tf.posit8
    elif sys.argv[1] == 'float16':
        eps = 1e-4
        posit = np.float16
        tf_type = tf.float16
    elif sys.argv[1] == 'float32':
        eps = 1e-8
        posit = np.float32
        tf_type = tf.float32
else:
    eps = 1e-8
    data_t = 'float32'
    posit = np.float32
    tf_type = tf.float32

# confirm dtype
print("\nType is: ", data_t)

# Normalize data
# X_train = (X_train/255.).astype(posit) # [0,1] normalization
# X_test = (X_test/255.).astype(posit)
X_train = ((X_train-127.5)/127.5).astype(posit)  # [-1,1] normalization
X_test = ((X_test-127.5)/127.5).astype(posit)

print("Input data type: {}".format(type(X_train[0, 0, 0, 0])))

"""## Implementation of LeNet-5
Implement the [LeNet-5](http://yann.lecun.com/exdb/lenet/) neural network architecture.

This is the only cell you need to edit.
# Input
The LeNet architecture accepts a 32x32xC image as input, where C is the number of color channels. Since MNIST images are grayscale, C is 1 in this case.

# Architecture
**Layer 1: Convolutional.** The output shape should be 28x28x6.

**Activation.** Your choice of activation function.

**Pooling.** The output shape should be 14x14x6.

**Layer 2: Convolutional.** The output shape should be 10x10x16.

**Activation.** Your choice of activation function.

**Pooling.** The output shape should be 5x5x16.

**Flatten.** Flatten the output shape of the final pooling layer such that it's 1D instead of 3D. The easiest way to do is by using `tf.contrib.layers.flatten`, which is already imported for you.

**Layer 3: Fully Connected.** This should have 120 outputs.

**Activation.** Your choice of activation function.

**Layer 4: Fully Connected.** This should have 84 outputs.

**Activation.** Your choice of activation function.

**Layer 5: Fully Connected (Logits).** This should have 10 outputs.

# Output
Return the result of the 2nd fully connected layer.
"""


def LeNet(x):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1

    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(
        shape=(5, 5, 1, 6), mean=mu, stddev=sigma, dtype=tf_type))
    conv1_b = tf.Variable(tf.zeros(6, dtype=tf_type))
    conv1 = tf.nn.conv2d(x, conv1_W,
                         strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # Activation.
    conv1 = tf.nn.relu(conv1)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='VALID')

    # Dropout
    #conv1 = tf.nn.dropout(conv1, keep_prob)

    # Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(
        shape=(5, 5, 6, 16), mean=mu, stddev=sigma, dtype=tf_type))
    conv2_b = tf.Variable(tf.zeros(16, dtype=tf_type))
    conv2 = tf.nn.conv2d(conv1, conv2_W,
                         strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    # Activation.
    conv2 = tf.nn.relu(conv2)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='VALID')

    # Dropout
    #conv2 = tf.nn.dropout(conv2, keep_prob)

    # Flatten. Input = 5x5x16. Output = 400.
    fc0 = flatten(conv2)
    #fc0 = tf.reshape(conv2, [-1])

    # Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(
        shape=(400, 120), mean=mu, stddev=sigma, dtype=tf_type))
    fc1_b = tf.Variable(tf.zeros(120, dtype=tf_type))
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b

    # Activation.
    fc1 = tf.nn.relu(fc1)

    # Dropout
    #fc1 = tf.nn.dropout(fc1, keep_prob)

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W = tf.Variable(tf.truncated_normal(
        shape=(120, 84), mean=mu, stddev=sigma, dtype=tf_type))
    fc2_b = tf.Variable(tf.zeros(84, dtype=tf_type))
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b

    # Activation.
    fc2 = tf.nn.relu(fc2)

    # Dropout
    #fc2 = tf.nn.dropout(fc2, keep_prob)

    # Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3_W = tf.Variable(tf.truncated_normal(
        shape=(84, 10), mean=mu, stddev=sigma, dtype=tf_type))
    fc3_b = tf.Variable(tf.zeros(10, dtype=tf_type))
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits


"""## Features and Labels
Train LeNet to classify [MNIST](http://yann.lecun.com/exdb/mnist/) data.

`x` is a placeholder for a batch of input images.
`y` is a placeholder for a batch of output labels.
"""

x = tf.placeholder(tf_type, (None, 32, 32, 1), name='inputs')
y = tf.placeholder(tf.int32, (None), name='labels')
#keep_prob = tf.placeholder(tf_type, (None))

"""## Training Pipeline
Create a training pipeline that uses the model to classify MNIST data.
"""

rate = posit(0.001)

logits = tf.identity(LeNet(x), name="logits")
# cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy, name="loss_operation")
optimizer = tf.train.AdamOptimizer(learning_rate=rate, beta1=posit(
    0.9), beta2=posit(0.999), epsilon=posit(eps))
training_operation = optimizer.minimize(
    loss_operation, name="training_operation")

# Create a dictionary to store activations
activations_dict = {}

"""## Model Evaluation
Evaluate how well the loss and accuracy of the model for a given dataset.
"""

# correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
correct_prediction = tf.equal(tf.argmax(logits, 1, output_type=tf.int32), y)
accuracy_operation = tf.reduce_mean(
    tf.cast(correct_prediction, tf.float32), name="accuracy_operation")
in_top5 = tf.nn.in_top_k(tf.cast(logits, tf.float32), y, k=5)
top5_operation = tf.reduce_mean(tf.cast(in_top5, tf.float32))
saver = tf.train.Saver()


def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    total_top5 = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset +
                                  BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy, top5 = sess.run([accuracy_operation, top5_operation], feed_dict={
                                  x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
        total_top5 += (top5 * len(batch_x))
    return (total_accuracy / num_examples, total_top5 / num_examples)


def get_top5(X_data, y_data):
    num_examples = len(X_data)
    total_top5 = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset +
                                  BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        top5 = sess.run(top5_operation, feed_dict={x: batch_x, y: batch_y})
        total_top5 += (top5 * len(batch_x))
    return (total_top5 / num_examples)


def validate(X_data, y_data):
    num_examples = len(X_data)
    total_loss = 0
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset +
                                  BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        _loss, _acc = sess.run([loss_operation, accuracy_operation], feed_dict={
                               x: batch_x, y: batch_y})
        total_accuracy += (_acc * len(batch_x))
        total_loss += (_loss * len(batch_x))
    return (total_loss / num_examples, total_accuracy / num_examples)

# Function to fetch activations from each layer
def fetch_activations(X_data, y_data, layer_name, sess):
    num_batches = len(X_data) // BATCH_SIZE
    activations = []

    for batch_index in range(num_batches):
        start = batch_index * BATCH_SIZE
        end = start + BATCH_SIZE
        batch_x, batch_y = X_data[start:end], y_data[start:end]

        batch_activations = sess.run(layer_name, feed_dict={x: batch_x, y: batch_y})
        activations.append(batch_activations)

    return activations

"""## Train the Model
Run the training data through the training pipeline to train the model.

Before each epoch, shuffle the training set.

After each epoch, measure the loss and accuracy of the validation set.

Save the model after training.
"""

hist = {}
# Adding list as value
hist["loss"] = []
hist["acc"] = []
hist["val_loss"] = []
hist["val_acc"] = []

files_path = './train_results/lenet5/' + data_set + '/'
directory = os.path.dirname(files_path)

if not os.path.exists(directory):
    os.makedirs(directory)

tic = time.time()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        total_train_accuracy = 0
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]

            _loss, _acc = sess.run([loss_operation, accuracy_operation], feed_dict={
                                   x: batch_x, y: batch_y})
            hist["loss"].append(_loss)
            hist["acc"].append(_acc)
            total_train_accuracy += (_acc * len(batch_x))

            # Fetch and save activations for all layers
            for layer_name in activations_dict.keys():
                layer_activations = fetch_activations(batch_x, batch_y, activations_dict[layer_name], sess)
                save_activations(layer_activations, layer_name, offset // BATCH_SIZE)

            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

        train_accuracy = total_train_accuracy / num_examples
        val_loss, val_accuracy = validate(X_test, y_test)
        hist["val_loss"].append(val_loss)
        hist["val_acc"].append(val_accuracy)

        print("EPOCH {} ...".format(i+1))
        print("Validation Loss = {:.3f}".format(val_loss))
        print("Validation Accuracy = {:.3f}".format(val_accuracy))
        print()

        if val_accuracy > 0.995:
            break

    saver.save(sess, './lenet/lenet')
    print("Model saved")

toc = time.time()
print(toc-tic)

# Save the loss history to a CSV file
save_layer_data("loss_history", hist)

# Save the activations history to CSV files
for layer_name in activations_dict.keys():
    activations_dir = f'./activation_data/{layer_name}'
    activations_files = sorted(os.listdir(activations_dir))
    all_activations = []
    for activations_file in activations_files:
        with open(os.path.join(activations_dir, activations_file), 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            activations = list(csv_reader)
            all_activations.extend(activations)
    with open(f'./activation_data/{layer_name}.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(all_activations)

print('Training is completed and activations are saved to CSV files.')

