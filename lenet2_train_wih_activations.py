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

# Set the start time before the session begins
tic = time.time()

def save_layer_data(layer_name, data):
    directory = './csv_data/'  # Set the directory for CSV files
    if not os.path.exists(directory):
        os.makedirs(directory)  # Create the directory if it does not exist
    csv_filename = f'{directory}{layer_name}.csv'
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        for layer in data:
            csv_writer.writerow(layer)

np.random.seed(1)
tf.set_random_seed(2)

if len(sys.argv) > 2:
    data_set = sys.argv[2]
    if data_set == 'mnist':
        d_set = mnist
    elif data_set == 'fashion_mnist':
        d_set = fashion_mnist
else:
    data_set = 'mnist'
    d_set = mnist

print("Dataset is: ", data_set)
(X_train, y_train), (X_test, y_test) = d_set.load_data()
X_train = np.expand_dims(X_train, axis=3)  # (60000, 28, 28, 1)
X_test = np.expand_dims(X_test, axis=3)  # (10000, 28, 28, 1)

print("\nImage Shape: {}\n".format(X_train[0].shape))
print("Training Set:   {} samples".format(len(X_train)))
print("Test Set:       {} samples".format(len(X_test)))

X_train = np.pad(X_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
X_test = np.pad(X_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

print("Updated Image Shape: {}".format(X_train[0].shape))

EPOCHS = 30
BATCH_SIZE = 128

if len(sys.argv) > 1:
    data_t = sys.argv[1]
    if data_t == 'posit32':
        eps = 1e-8
        posit = np.posit32
        tf_type = tf.posit32
    elif data_t == 'posit16':
        eps = 1e-4
        posit = np.posit16
        tf_type = tf.posit16
    elif data_t == 'posit8':
        eps = 0.015625
        posit = np.posit8
        tf_type = tf.posit8
    elif data_t == 'float16':
        eps = 1e-4
        posit = np.float16
        tf_type = tf.float16
    elif data_t == 'float32':
        eps = 1e-8
        posit = np.float32
        tf_type = tf.float32
else:
    eps = 1e-8
    data_t = 'float32'
    posit = np.float32
    tf_type = tf.float32

X_train = ((X_train-127.5)/127.5).astype(posit)
X_test = ((X_test-127.5)/127.5).astype(posit)

x = tf.placeholder(tf_type, (None, 32, 32, 1), name='inputs')
y = tf.placeholder(tf.int32, (None), name='labels')

def LeNet(x):
    mu = 0
    sigma = 0.1
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean=mu, stddev=sigma, dtype=tf_type))
    conv1_b = tf.Variable(tf.zeros(6, dtype=tf_type))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    conv1 = tf.nn.relu(conv1)
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma, dtype=tf_type))
    conv2_b = tf.Variable(tf.zeros(16, dtype=tf_type))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    conv2 = tf.nn.relu(conv2)
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    fc0 = flatten(conv2)
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma, dtype=tf_type))
    fc1_b = tf.Variable(tf.zeros(120, dtype=tf_type))
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b
    fc1 = tf.nn.relu(fc1)

    fc2_W = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma, dtype=tf_type))
    fc2_b = tf.Variable(tf.zeros(84, dtype=tf_type))
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b
    fc2 = tf.nn.relu(fc2)

    fc3_W = tf.Variable(tf.truncated_normal(shape=(84, 10), mean=mu, stddev=sigma, dtype=tf_type))
    fc3_b = tf.Variable(tf.zeros(10, dtype=tf_type))
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits, conv1, conv2, fc1, fc2  # Return intermediate activations for analysis

rate = posit(0.001)
logits, conv1, conv2, fc1, fc2 = LeNet(x)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy, name="loss_operation")
optimizer = tf.train.AdamOptimizer(learning_rate=rate, beta1=posit(0.9), beta2=posit(0.999), epsilon=posit(eps))
training_operation = optimizer.minimize(loss_operation, name="training_operation")

correct_prediction = tf.equal(tf.argmax(logits, 1, output_type=tf.int32), y)
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy_operation")
in_top5 = tf.nn.in_top_k(tf.cast(logits, tf.float32), y, k=5)
top5_operation = tf.reduce_mean(tf.cast(in_top5, tf.float32))
saver = tf.train.Saver()

def validate(X_data, y_data):
    num_examples = len(X_data)
    total_loss = 0
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
        _loss, _acc = sess.run([loss_operation, accuracy_operation], feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (_acc * len(batch_x))
        total_loss += (_loss * len(batch_x))
    return (total_loss / num_examples, total_accuracy / num_examples)

files_path = './train_results/lenet5/' + data_set + '/'
if not os.path.exists(files_path):
    os.makedirs(files_path)

results_path = files_path + 'training_results.csv'  # Path where the training results will be saved

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    print("Training...")
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            _, _loss, _acc, act_conv1, act_conv2, act_fc1, act_fc2 = sess.run(
                [training_operation, loss_operation, accuracy_operation, conv1, conv2, fc1, fc2],
                feed_dict={x: batch_x, y: batch_y})
            
            # Save activations for each epoch
            save_layer_data('conv1_epoch_{}'.format(i), act_conv1.tolist())
            save_layer_data('conv2_epoch_{}'.format(i), act_conv2.tolist())
            save_layer_data('fc1_epoch_{}'.format(i), act_fc1.tolist())
            save_layer_data('fc2_epoch_{}'.format(i), act_fc2.tolist())

        val_loss, val_acc = validate(X_test, y_test)
        print("Epoch {} training complete. Loss: {:.4f}, Accuracy: {:.4f}, Validation Loss: {:.4f}, Validation Accuracy: {:.4f}".format(
            i+1, _loss, _acc, val_loss, val_acc))

    # Save the model
    model_name = files_path + data_t + '.ckpt'
    save_path = saver.save(sess, model_name)
    print("Model saved in path: %s" % save_path)

def get_top5(X_data, y_data):
    num_examples = len(X_data)
    total_top5 = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
        top5 = sess.run(top5_operation, feed_dict={x: batch_x, y: batch_y})
        total_top5 += (top5 * len(batch_x))
    return total_top5 / num_examples

with tf.Session as sess:
    saver.restore(sess, model_name)
    test_top5 = get_top5(X_test, y_test)
    print("Test Top-5 Accuracy: {:.4f}".format(test_top5))

# Output overall training time and result summary
toc = time.time()
elapsed_time = toc - tic
print("Training completed in {:.2f} seconds. Detailed performance metrics saved to {}".format(elapsed_time, results_path))
