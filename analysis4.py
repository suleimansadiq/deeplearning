import tensorflow as tf
import csv
import numpy as np
import os

def main():
    # Define the same model architecture as in your training script
    def LeNet(x):
        # Your LeNet architecture definition here...

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

        # Create output folder if it doesn't exist
        output_folder = 'version2.0'
        os.makedirs(output_folder, exist_ok=True)

        # Export the variables to CSV files in the output folder
        for variable_name, variable_tensor in variables_to_export.items():
            values = sess.run(variable_tensor)
            export_to_csv(output_folder, variable_name, values)

def export_to_csv(output_folder, variable_name, values):
    # Specify the CSV file name with the output folder path
    csv_filename = f'{output_folder}/{variable_name}.csv'

    # Open the CSV file for writing
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write the values to the CSV file based on their shape
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
