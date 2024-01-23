import tensorflow as tf

def main():
    # Define the same model architecture as in your training script
    def LeNet(x):
        # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
        mu = 0
        sigma = 0.1

        # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
        conv1_W = tf.Variable(tf.truncated_normal(
            shape=(5, 5, 1, 6), mean=mu, stddev=sigma, dtype=tf.posit8))
        conv1_b = tf.Variable(tf.zeros(6, dtype=tf.posit8))
        conv1 = tf.nn.conv2d(x, conv1_W,
                             strides=[1, 1, 1, 1], padding='VALID') + conv1_b

        # Activation.
        conv1 = tf.nn.relu(conv1)

        # Pooling. Input = 28x28x6. Output = 14x14x6.
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1], padding='VALID')

        # Dropout
        # conv1 = tf.nn.dropout(conv1, keep_prob)

        # Layer 2: Convolutional. Output = 10x10x16.
        conv2_W = tf.Variable(tf.truncated_normal(
            shape=(5, 5, 6, 16), mean=mu, stddev=sigma, dtype=tf.posit8))
        conv2_b = tf.Variable(tf.zeros(16, dtype=tf.posit8))
        conv2 = tf.nn.conv2d(conv1, conv2_W,
                             strides=[1, 1, 1, 1], padding='VALID') + conv2_b

        # Activation.
        conv2 = tf.nn.relu(conv2)

        # Pooling. Input = 10x10x16. Output = 5x5x16.
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1], padding='VALID')

        # Dropout
        # conv2 = tf.nn.dropout(conv2, keep_prob)

        # Flatten. Input = 5x5x16. Output = 400.
        fc0 = flatten(conv2)
        # fc0 = tf.reshape(conv2, [-1])

        # Layer 3: Fully Connected. Input = 400. Output = 120.
        fc1_W = tf.Variable(tf.truncated_normal(
            shape=(400, 120), mean=mu, stddev=sigma, dtype=tf.posit8))
        fc1_b = tf.Variable(tf.zeros(120, dtype=tf.posit8))
        fc1 = tf.matmul(fc0, fc1_W) + fc1_b

        # Activation.
        fc1 = tf.nn.relu(fc1)

        # Dropout
        # fc1 = tf.nn.dropout(fc1, keep_prob)

        # Layer 4: Fully Connected. Input = 120. Output = 84.
        fc2_W = tf.Variable(tf.truncated_normal(
            shape=(120, 84), mean=mu, stddev=sigma, dtype=tf.posit8))
        fc2_b = tf.Variable(tf.zeros(84, dtype=tf.posit8))
        fc2 = tf.matmul(fc1, fc2_W) + fc2_b

        # Activation.
        fc2 = tf.nn.relu(fc2)

        # Dropout
        # fc2 = tf.nn.dropout(fc2, keep_prob)

        # Layer 5: Fully Connected. Input = 84. Output = 10.
        fc3_W = tf.Variable(tf.truncated_normal(
            shape=(84, 10), mean=mu, stddev=sigma, dtype=tf.posit8))
        fc3_b = tf.Variable(tf.zeros(10, dtype=tf.posit8))
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

        # Print the values of weights and biases
        print("conv1_W:", sess.run(conv1_W))
        print("conv2_W:", sess.run(conv2_W))
        print("fc1_W:", sess.run(fc1_W))
        print("fc2_W:", sess.run(fc2_W))
        print("fc3_W:", sess.run(fc3_W))

        print("conv1_b:", sess.run(conv1_b))
        print("conv2_b:", sess.run(conv2_b))
        print("fc1_b:", sess.run(fc1_b))
        print("fc2_b:", sess.run(fc2_b))
        print("fc3_b:", sess.run(fc3_b))

if __name__ == "__main__":
    main()
