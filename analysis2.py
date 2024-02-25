import tensorflow as tf
import csv
import numpy as np

def main():
    # Define the same model architecture as in your training script
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

    # Create a TensorFlow session
    with tf.Session() as sess:
        # Load the trained model checkpoint
        saver = tf.train.import_meta_graph('posit8.ckpt.meta')  
        saver.restore(sess, 'posit8.ckpt') 

        # Access the variables from the loaded model
        graph = tf.get_default_graph()
        conv1_W = graph.get_tensor_by_name('Variable:0')  
        conv2_W = graph.get_tensor_by_name('Variable_1:0')  
        fc1_W = graph.get_tensor_by_name('Variable_2:0')  
        fc2_W = graph.get_tensor_by_name('Variable_3:0')  
        fc3_W = graph.get_tensor_by_name('Variable_4:0')  

        conv1_b = graph.get_tensor_by_name('Variable_5:0')  
        conv2_b = graph.get_tensor_by_name('Variable_6:0')  
        fc1_b = graph.get_tensor_by_name('Variable_7:0')  
        fc2_b = graph.get_tensor_by_name('Variable_8:0')  
        fc3_b = graph.get_tensor_by_name('Variable_9:0')  

        # Create a dictionary with variable names and corresponding tensors
        variables_to_export = {
            'conv1_W': conv1_W,
            'conv2_W': conv2_W,
            'fc1_W': fc1_W,
            'fc2_W': fc2_W,
            'fc3_W': fc3_W,
            'conv1_b': conv1_b,
            'conv2_b': conv2_b,
            'fc1_b': fc1_b,
            'fc2_b': fc2_b,
            'fc3_b': fc3_b,
        }

        # Export the variables to CSV files
        for variable_name, variable_tensor in variables_to_export.items():
            values = sess.run(variable_tensor)
            export_to_csv(variable_name, values)

def export_to_csv(variable_name, values):
    # Specify the CSV file name
    csv_filename = f'{variable_name}.csv'

    # Open the CSV file for writing
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write a header row with labels
        if len(values.shape) == 1:
            writer.writerow(['Index', 'Value'])
            for index, value in enumerate(values):
                writer.writerow([index, value])
        elif len(values.shape) == 2:
            writer.writerow(['Row', 'Column', 'Value'])
            for row in range(values.shape[0]):
                for col in range(values.shape[1]):
                    writer.writerow([row, col, values[row, col]])
        elif len(values.shape) == 4:
            writer.writerow(['Filter', 'Row', 'Column', 'Depth', 'Value'])
            for filter_idx in range(values.shape[3]):
                for row in range(values.shape[0]):
                    for col in range(values.shape[1]):
                        for depth in range(values.shape[2]):
                            value = values[row, col, depth, filter_idx]
                            writer.writerow([filter_idx, row, col, depth, value])

    print(f'{variable_name} values have been exported to {csv_filename}.')

if __name__ == "__main__":
    main()
