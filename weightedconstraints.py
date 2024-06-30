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
    def convert_to_weighted_linear_sum_constraints(weights, biases):
        constraints = []
        if len(weights.shape) == 4:  # Convolutional layer
            filter_height, filter_width, input_depth, num_filters = weights.shape
            for n in range(num_filters):
                for fh in range(filter_height):
                    for fw in range(filter_width):
                        for fd in range(input_depth):
                            constraint = f"{biases[n]} + {weights[fh, fw, fd, n]}*x{fh}{fw}{fd} >= 0"
                            constraints.append(constraint)
                            relu_constraint = f"y_out = if ({biases[n]} + {weights[fh, fw, fd, n]}*x{fh}{fw}{fd}) >= 0 then ({biases[n]} + {weights[fh, fw, fd, n]}*x{fh}{fw}{fd}) else 0"
                            constraints.append(relu_constraint)
        else:  # Fully connected layer
            for i in range(len(biases)):
                constraint = f"{biases[i]} + " + " + ".join([f"{weights[j, i]}*x{j}" for j in range(weights.shape[0])]) + " >= 0"
                constraints.append(constraint)
                relu_constraint = f"y_out = if ({biases[i]} + " + " + ".join([f"{weights[j, i]}*x{j}" for j in range(weights.shape[0])]) + f") >= 0 then ({biases[i]} + " + " + ".join([f"{weights[j, i]}*x{j}" for j in range(weights.shape[0])]) + ") else 0"
                constraints.append(relu_constraint)
        return constraints

    all_constraints = []
    layer_pairs = [
        ('Variable:0', 'Variable_1:0'),
        ('Variable_2:0', 'Variable_3:0'),
        ('Variable_4:0', 'Variable_5:0'),
        ('Variable_6:0', 'Variable_7:0'),
        ('Variable_8:0', 'Variable_9:0')
    ]

    for weight_name, bias_name in layer_pairs:
        weights = weights_and_biases[weight_name]
        biases = weights_and_biases[bias_name]
        constraints = convert_to_weighted_linear_sum_constraints(weights, biases)
        print(f"Generated {len(constraints)} constraints for layer {weight_name} and {bias_name}")
        all_constraints.extend(constraints)

# Save constraints to a file
output_file = './weighted_linear_sum_constraints.txt'
with open(output_file, 'w') as f:
    for constraint in all_constraints:
        f.write(f'{constraint}\n')

print(f'Total number of constraints: {len(all_constraints)}')
print(f'Constraints saved to {output_file}')
