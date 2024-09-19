#include <stdio.h>
#include <stdlib.h>
#include <softposit.h>  // Include SoftPosit library

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

// Function to convert a double to an 8-bit posit and return its bitwise string representation
char* convertToPosit8BitString(double constant_value) {
    // Convert the double to an 8-bit posit
    posit8_t posit_value = convertDoubleToP8(constant_value);
    
    // Convert the posit to a bitwise string (8-bit)
    char *posit_str = positToBitString(posit_value);
    return posit_str;
}

int main(void) {
    double input_value;
    char continue_choice;

    do {
        // Prompt user for input
        printf("Enter a double value to convert to an 8-bit posit: ");
        scanf("%lf", &input_value);

        // Convert to posit and display result
        char *posit_str = convertToPosit8BitString(input_value);
        printf("8-bit posit equivalent: %s\n", posit_str);

        // Free allocated memory
        free(posit_str);

        // Ask if the user wants to continue
        printf("Do you want to convert another value? (y/n): ");
        scanf(" %c", &continue_choice);

    } while (continue_choice == 'y' || continue_choice == 'Y');

    printf("Exiting...\n");
    return 0;
}
