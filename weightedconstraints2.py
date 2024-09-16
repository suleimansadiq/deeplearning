import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.contrib.layers import flatten
import sys

# Load the modified TensorFlow with posit support
if len(sys.argv) > 1:
    data_t = sys.argv[1]
    if data_t == 'posit32':
        posit = np.posit32
        tf_type = tf.posit32
    elif data_t == 'posit16':
        posit = np.posit16
        tf_type = tf.posit16
    elif data_t == 'posit8':
        posit = np.posit8
        tf_type = tf.posit8
    elif data_t == 'float16':
        posit = np.float16
        tf_type = tf.float16
    elif data_t == 'float32':
        posit = np.float32
        tf_type = tf.float32
else:
    data_t = 'float32'
    posit = np.float32
    tf_type = tf.float32

# Define the LeNet architecture with explicit variable names
def LeNet(x):
    mu = 0
    sigma = 0.1

    conv1_W = tf.Variable(tf.truncated_normal(
        shape=(5, 5, 1, 6), mean=mu, stddev=sigma, dtype=tf_type), name='Variable')
    conv1_b = tf.Variable(tf.zeros(6, dtype=tf_type), name='Variable_1')
    conv1 = tf.nn.conv2d(x, conv1_W,
                         strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    conv1 = tf.nn.relu(conv1)
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='VALID')

    conv2_W = tf.Variable(tf.truncated_normal(
        shape=(5, 5, 6, 16), mean=mu, stddev=sigma, dtype=tf_type), name='Variable_2')
    conv2_b = tf.Variable(tf.zeros(16, dtype=tf_type), name='Variable_3')
    conv2 = tf.nn.conv2d(conv1, conv2_W,
                         strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    conv2 = tf.nn.relu(conv2)
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='VALID')

    fc0 = flatten(conv2)

    fc1_W = tf.Variable(tf.truncated_normal(
        shape=(400, 120), mean=mu, stddev=sigma, dtype=tf_type), name='Variable_4')
    fc1_b = tf.Variable(tf.zeros(120, dtype=tf_type), name='Variable_5')
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b
    fc1 = tf.nn.relu(fc1)

    fc2_W = tf.Variable(tf.truncated_normal(
        shape=(120, 84), mean=mu, stddev=sigma, dtype=tf_type), name='Variable_6')
    fc2_b = tf.Variable(tf.zeros(84, dtype=tf_type), name='Variable_7')
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b
    fc2 = tf.nn.relu(fc2)

    fc3_W = tf.Variable(tf.truncated_normal(
        shape=(84, 10), mean=mu, stddev=sigma, dtype=tf_type), name='Variable_8')
    fc3_b = tf.Variable(tf.zeros(10, dtype=tf_type), name='Variable_9')
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits

# Define placeholders and model
x = tf.placeholder(tf_type, (None, 32, 32, 1), name='inputs')
y = tf.placeholder(tf.int32, (None), name='labels')

logits = LeNet(x)

saver = tf.train.Saver()

# Load the trained model
model_checkpoint_path = './posit8.ckpt'

with tf.Session() as sess:
    saver.restore(sess, model_checkpoint_path)

    # Extract weights and biases
    weights_and_biases = {}
    for var in tf.trainable_variables():
        weights_and_biases[var.name] = sess.run(var)
        print(f"Loaded variable {var.name} with shape {sess.run(var).shape}")

    # Convert to weighted linear sum constraints
    def convert_to_weighted_linear_sum_constraints(weights, biases, input_shape, layer_name, constraint_index, prev_layer_name=None):
        constraints = []
        node_count = 0  # To count the actual number of nodes

        if len(weights.shape) == 4:  # Convolutional layer
            filter_height, filter_width, input_depth, num_filters = weights.shape
            for n in range(num_filters):
                for i in range(input_shape[0] - filter_height + 1):
                    for j in range(input_shape[1] - filter_width + 1):
                        linear_sum = f"{biases[n]}"
                        for fh in range(filter_height):
                            for fw in range(filter_width):
                                for fd in range(input_depth):
                                    input_var = f"z_{prev_layer_name}_{i+fh}_{j+fw}_{fd}" if prev_layer_name else f"x_{i+fh}_{j+fw}_{fd}"
                                    linear_sum += f" + {weights[fh, fw, fd, n]}*{input_var}"
                        constraints.append(f"{constraint_index}: y_{layer_name}_{i}_{j}_{n} = {linear_sum}")
                        constraint_index += 1
                        constraints.append(f"{constraint_index}: z_{layer_name}_{i}_{j}_{n} = if y_{layer_name}_{i}_{j}_{n} >= 0 then y_{layer_name}_{i}_{j}_{n} else 0")
                        constraint_index += 1
                        node_count += 1
        else:  # Fully connected layer (handle 1D input)
            input_size = np.prod(input_shape)  # Flattened size
            for i in range(len(biases)):
                linear_sum = f"{biases[i]}"
                for j in range(input_size):
                    input_var = f"z_{prev_layer_name}_{j}"  # Use flat index for fully connected layers
                    if j < weights.shape[0]:
                        linear_sum += f" + {weights[j, i]}*{input_var}"
                constraints.append(f"{constraint_index}: y_{layer_name}_{i} = {linear_sum}")
                constraint_index += 1
                constraints.append(f"{constraint_index}: z_{layer_name}_{i} = if y_{layer_name}_{i} >= 0 then y_{layer_name}_{i} else 0")
                constraint_index += 1
                node_count += 1

        return constraints, node_count, constraint_index

    # Max-Pooling representation for constraints
    def apply_max_pooling_exactly(input_shape, prev_layer_name, layer_name, constraint_index):
        # Just perform the pooling but do not generate constraints for it
        return [], 0, constraint_index  # Returning empty constraints for pooling

    all_constraints = []
    input_shape = (32, 32, 1)  # Initial input shape
    constraint_index = 1  # To index the constraints

    layer_pairs = [
        ('Variable:0', 'Variable_1:0'),
        ('Variable_2:0', 'Variable_3:0'),
        ('Variable_4:0', 'Variable_5:0'),
        ('Variable_6:0', 'Variable_7:0'),
        ('Variable_8:0', 'Variable_9:0')
    ]

    layer_names = ["conv1", "conv2", "fc1", "fc2", "fc3"]
    total_constraints = 0
    total_nodes = 0  # To count the total number of nodes

    processed_layers = set()  # Set to track processed layers
    prev_layer_name = None

    for (weight_name, bias_name), layer_name in zip(layer_pairs, layer_names):
        weights = weights_and_biases[weight_name]
        biases = weights_and_biases[bias_name]

        # Ensure each layer is processed only once
        if layer_name in processed_layers:
            continue

        # Generate constraints for each layer
        constraints, node_count, constraint_index = convert_to_weighted_linear_sum_constraints(
            weights, biases, input_shape, layer_name, constraint_index, prev_layer_name)
        num_constraints = len(constraints)
        print(f"Generated {num_constraints} constraints for layer {layer_name}")
        all_constraints.extend(constraints)
        total_constraints += num_constraints  # Total number of constraints (each node has 2 constraints)
        total_nodes += node_count  # Total number of nodes

        # Mark the current layer as processed
        processed_layers.add(layer_name)

        # Update input shape for the next layer based on the current layer output
        if layer_name == "conv1":
            input_shape = (14, 14, 6)  # Output size after conv1 max-pooling
        elif layer_name == "conv2":
            input_shape = (5, 5, 16)  # Output size after conv2 max-pooling
        elif layer_name in ["fc1", "fc2", "fc3"]:
            input_shape = (weights.shape[1],)  # Flattened size for fully connected layers

        prev_layer_name = layer_name

    # Save constraints to file
    with open('weighted_linear_sum_constraints2.txt', 'w') as f:
        for constraint in all_constraints:
            f.write(f"{constraint}\n\n")  # Adding a blank line between each constraint

    print(f"Total number of nodes: {total_nodes}")
    print(f"Total number of constraints: {total_constraints}")
