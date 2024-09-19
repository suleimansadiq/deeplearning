#include <stdio.h>
#include <stdlib.h>
#include <regex.h>
#include <string.h>
#include "softposit.h"  // Include SoftPosit library

// Function to convert an 8-bit posit to a string of 0s and 1s
char* positToBitString(posit8_t posit_value) {
    char *bit_str = (char*) malloc(9 * sizeof(char));  // 8 bits + 1 for null terminator
    if (bit_str == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    // Extract each bit of the posit and convert to '0' or '1'
    for (int i = 0; i < 8; i++) {
        bit_str[7 - i] = (posit_value.v & (1 << i)) ? '1' : '0';  // Extract each bit from the posit
    }
    bit_str[8] = '\0';  // Null-terminate the string
    return bit_str;
}

// Function to convert a constant string to an 8-bit posit and return its bitwise string representation
char* convertToPosit8BitString(char* constant_str) {
    // Convert the constant string to a double
    double constant_value = atof(constant_str);
    printf("Original constant: %f\n", constant_value);  // Log the original constant
    
    // Convert the double to an 8-bit posit
    posit8_t posit_value = convertDoubleToP8(constant_value);
    printf("Converted posit value (hex): 0x%x\n", posit_value.v);  // Log the posit value in hex
    
    // Convert the posit to a bitwise string (8-bit)
    char *posit_str = positToBitString(posit_value);
    printf("Converted posit value (bit string): %s\n", posit_str);  // Log the bitwise representation
    
    return posit_str;
}

int main(void) {
    // File paths
    char* input_file_path = "/Users/suleimansadiq/Documents/Codes/PositToBDD/weighted_linear_sum_constraints2.txt";
    char* output_file_path = "/Users/suleimansadiq/Documents/Codes/PositToBDD/weighted_linear_sum_constraints_posits.txt";
    
    // Open the input file for reading
    FILE *input_file = fopen(input_file_path, "r");
    if (input_file == NULL) {
        fprintf(stderr, "Error opening input file\n");
        return 1;
    }

    // Open the output file for writing
    FILE *output_file = fopen(output_file_path, "w");
    if (output_file == NULL) {
        fprintf(stderr, "Error opening output file\n");
        fclose(input_file);
        return 1;
    }

    // Regex pattern to match valid floating-point constants (with optional signs and decimals)
    char* regex_pattern = "^[-+]?[0-9]*\\.?[0-9]+$";  // Strictly match constants without variables

    // Regex pattern to separate constants from expressions like "0.015625*x_0_1_0"
    char* extract_constant_pattern = "^([-+]?[0-9]*\\.?[0-9]+)(.*)";  // Extract constant and remaining expression

    regex_t regex, extract_regex;
    regcomp(&regex, regex_pattern, REG_EXTENDED);
    regcomp(&extract_regex, extract_constant_pattern, REG_EXTENDED);

    // Read each line from the input file
    char line[1024];
    int constraint_count = 0;

    while (fgets(line, sizeof(line), input_file)) {
        if (constraint_count >= 2) {
            break;  // Process only the first two constraints
        }

        char final_line[2048];  // Buffer to hold the final modified line
        final_line[0] = '\0';   // Initialize empty string for each line
        char *current_position = line;

        printf("Processing line: %s", line);  // Log the current line being processed

        // Process constraint index first (e.g., "1:")
        char *constraint_number_end = strchr(current_position, ':');
        if (constraint_number_end != NULL) {
            // Copy the constraint number and ":" directly into the final line
            int index_len = (int)(constraint_number_end - current_position + 1);
            strncat(final_line, current_position, index_len);
            strcat(final_line, " ");
            current_position = constraint_number_end + 1;  // Move past the constraint index
        }

        // Tokenize the remaining part by spaces to process each component
        char *token = strtok(current_position, " ");
        while (token != NULL) {
            printf("Token: %s\n", token);  // Log each token

            // Check if the token is a pure constant (no variables)
            if (regexec(&regex, token, 0, NULL, 0) == 0) {
                printf("Constant found: %s\n", token);  // Log that a constant was found
                char *posit_str = convertToPosit8BitString(token);
                strcat(final_line, posit_str);  // Append the posit string to the final line
                free(posit_str);  // Free the memory allocated for the posit string
            }
            // Check if the token contains a constant and a variable (like "0.015625*x_0_1_0")
            else if (regexec(&extract_regex, token, 0, NULL, 0) == 0) {
                printf("Constant with variable found: %s\n", token);  // Log the complex token
                regmatch_t matches[3];
                if (regexec(&extract_regex, token, 3, matches, 0) == 0) {
                    // Extract the constant part
                    char constant_str[100];
                    int match_length = (int)(matches[1].rm_eo - matches[1].rm_so + 1);  // Declare an int to store the length
                    match_length = (int)((matches[1].rm_eo) - (matches[1].rm_so) + 1);  // Explicit cast to int
                    snprintf(constant_str, match_length, "%s", token + matches[1].rm_so);  // Use match_length



                    // Extract the remaining part (variable with operator)
                    char remaining_str[100];
                    snprintf(remaining_str, matches[2].rm_eo - matches[2].rm_so + 1, "%s", token + matches[2].rm_so);

                    // Convert the constant to posit
                    char *posit_str = convertToPosit8BitString(constant_str);
                    strcat(final_line, posit_str);  // Append the posit string
                    strcat(final_line, remaining_str);  // Append the remaining expression (variable)
                    free(posit_str);  // Free the posit string memory
                }
            }
            else {
                printf("Variable or operator: %s\n", token);  // Log variable or operator
                strcat(final_line, token);  // Append as-is
            }

            // Add a space after each token (except for the last one)
            token = strtok(NULL, " ");
            if (token != NULL) {
                strcat(final_line, " ");
            }
        }

        // Write the final modified line to the output file
        printf("Final processed line: %s\n", final_line);  // Log the final output
        fputs(final_line, output_file);

        // Increment constraint count
        constraint_count++;
    }

    // Clean up
    regfree(&regex);
    regfree(&extract_regex);
    fclose(input_file);
    fclose(output_file);

    printf("Successfully converted constants to 8-bit posit for first two constraints and saved to %s\n", output_file_path);
    return 0;
}
