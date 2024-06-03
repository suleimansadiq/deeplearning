import tensorflow as tf
import csv
import numpy as np
import os

def main():
    # Define the same model architecture as in your training script
    def LeNet(x):
        # Your LeNet architecture definition here...
        # Assuming you have defined the layers and activations
        
        # Create a TensorFlow session
        with tf.Session() as sess:
            # Load the trained model checkpoint
            saver = tf.train.import_meta_graph('posit8.ckpt.meta')  
            saver.restore(sess, 'posit8.ckpt') 

            # Access the variables from the loaded model
            graph = tf.get_default_graph()

            # Initialize weighted linear sums and biases
            weighted_linear_sums = []
            biases = []
            
            # Initialize the first weighted linear sum with bias
            weighted_linear_sum = graph.get_tensor_by_name('Variable:0')  # Assuming first layer's weight
            bias = graph.get_tensor_by_name('Variable_5:0')  # Assuming first layer's bias
            weighted_linear_sums.append(weighted_linear_sum)
            biases.append(bias)
            
            # Adjust the weighted linear sums for subsequent layers
            for layer_idx in range(1, num_layers):  # Assuming you have num_layers defined
                weighted_linear_sum = tf.matmul(weighted_linear_sum, weights[layer_idx]) + biases[layer_idx]
                weighted_linear_sums.append(weighted_linear_sum)

        # Export the weighted linear sums and biases to CSV files
        output_folder = 'weighted'
        os.makedirs(output_folder, exist_ok=True)

        for idx, wls in enumerate(weighted_linear_sums):
            export_to_csv(output_folder, f'weighted_linear_sum_{idx}', sess.run(wls))
        for idx, bias in enumerate(biases):
            export_to_csv(output_folder, f'bias_{idx}', sess.run(bias))

    print("Main function called.")
    main()

def export_to_csv(output_folder, variable_name, values):
    print(f'Exporting {variable_name} to CSV...')
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
    print("Script started.")
    main()
    print("Script completed.")
