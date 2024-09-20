#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cudd.h"

#define INITIAL_VAR_CAPACITY 1000  // Initial capacity for the variable mapping

typedef struct {
    char variable_name[100];
    DdNode* bdd_var;
} VariableMapping;

DdNode* getOrCreateBDDVar(DdManager* manager, VariableMapping** mappings, int* var_count, int* var_capacity, char* variable_str) {
    // Check for 'max(' and remove it from the variable string
    if (strncmp(variable_str, "max(", 4) == 0) {
        variable_str += 4; // Skip "max("
        char* closing_bracket = strchr(variable_str, ')');
        if (closing_bracket != NULL) {
            *closing_bracket = '\0';  // Null-terminate the string
        }
    }

    // Search for existing variable
    for (int i = 0; i < *var_count; i++) {
        if (strcmp((*mappings)[i].variable_name, variable_str) == 0) {
            printf("Variable %s already exists, using existing BDD variable.\n", variable_str);
            return (*mappings)[i].bdd_var;
        }
    }

    // If we've reached the capacity, dynamically increase the size
    if (*var_count >= *var_capacity) {
        *var_capacity *= 2;  // Double the capacity
        *mappings = (VariableMapping*) realloc(*mappings, (*var_capacity) * sizeof(VariableMapping));
        if (*mappings == NULL) {
            fprintf(stderr, "Error: Failed to reallocate memory for variable mappings.\n");
            exit(1);
        }
        printf("Resized the variable mapping array to %d.\n", *var_capacity);
    }

    // Create a new BDD variable
    DdNode* new_var = Cudd_bddNewVar(manager);
    Cudd_Ref(new_var);

    // Store the new variable in the mappings
    strcpy((*mappings)[*var_count].variable_name, variable_str);
    (*mappings)[*var_count].bdd_var = new_var;
    (*var_count)++;

    printf("Created new BDD variable for %s, var_count is now %d\n", variable_str, *var_count);
    return new_var;
}

void processConstraintsFromFile(DdManager* manager, VariableMapping** mappings, int* var_count, int* var_capacity, const char* file_path) {
    FILE* file = fopen(file_path, "r");
    if (!file) {
        fprintf(stderr, "Error opening file %s\n", file_path);
        exit(1);
    }

    char line[1024];
    while (fgets(line, sizeof(line), file)) {
        printf("Processing constraint: %s", line);

        char* token = strtok(line, " ");
        while (token != NULL) {
            // Skip equal sign or any other non-variable token
            if (strcmp(token, "=") == 0 || strcmp(token, "+") == 0) {
                token = strtok(NULL, " ");
                continue;
            }

            // Extract variable and check if it is an 'x_', 'y_', 'z_', or 'm_' type variable
            if (strstr(token, "x_") || strstr(token, "y_") || strstr(token, "z_") || strstr(token, "m_")) {
                DdNode* bdd_var = getOrCreateBDDVar(manager, mappings, var_count, var_capacity, token);
                // In this context, you could add the BDD variables into a larger equation or decision diagram
            }

            token = strtok(NULL, " ");
        }
    }

    fclose(file);
}

int main(void) {
    DdManager* manager = Cudd_Init(0, 0, CUDD_UNIQUE_SLOTS, CUDD_CACHE_SLOTS, 0);

    // Dynamic allocation for the variable mapping array
    int var_count = 0;
    int var_capacity = INITIAL_VAR_CAPACITY;
    VariableMapping* mappings = (VariableMapping*) malloc(var_capacity * sizeof(VariableMapping));

    if (mappings == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for variable mappings.\n");
        return 1;
    }

    // Path to your constraints file
    const char* file_path = "/Users/suleimansadiq/Documents/Codes/PositToBDD/weighted_linear_sum_constraints_posits.txt";

    // Process constraints from the file and map them to BDD variables
    processConstraintsFromFile(manager, &mappings, &var_count, &var_capacity, file_path);

    // Cleanup
    free(mappings);
    Cudd_Quit(manager);

    printf("Finished processing BDDs for constraints.\n");
    return 0;
}
