#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <cudd.h>

#define MAX_LINE_LENGTH 1024

// Function to create BDDs from constraints
DdNode* createBDDFromConstraint(DdManager* manager, char* constraint, char* inputs) {
    char* token;
    DdNode* bdd = NULL;
    DdNode* temp = NULL;
    CUDD_VALUE_TYPE value;

    // Tokenize the constraint
    token = strtok(constraint, " ");
    while (token != NULL) {
        if (strstr(token, "x_") != NULL || strstr(token, "y_") != NULL || strstr(token, "z_") != NULL) {
            int varIndex = atoi(token + 2); // Get the variable index (assuming variables are like x_i, y_i, z_i)
            DdNode* var = Cudd_bddIthVar(manager, varIndex);
            Cudd_Ref(var);

            // Check for negation
            if (strstr(token, "-") != NULL) {
                var = Cudd_Not(var);
                Cudd_Ref(var);
            }

            if (bdd == NULL) {
                bdd = var;
            } else {
                temp = Cudd_bddAnd(manager, bdd, var);
                Cudd_Ref(temp);
                Cudd_RecursiveDeref(manager, bdd);
                bdd = temp;
            }

            // Add variable to inputs string
            char varString[20];
            snprintf(varString, sizeof(varString), "%s ", token);
            strcat(inputs, varString);
        }
        token = strtok(NULL, " ");
    }

    if (bdd == NULL) {
        bdd = Cudd_ReadOne(manager); // Default to constant 1 if no variables are found
    }
    Cudd_Ref(bdd);
    return bdd;
}

int main() {
    FILE *file;
    char line[MAX_LINE_LENGTH];
    int constraint_count = 0;

    // Initialize CUDD
    DdManager* manager = Cudd_Init(0, 0, CUDD_UNIQUE_SLOTS, CUDD_CACHE_SLOTS, 0);

    file = fopen("/Users/suleimansadiq/Documents/Codes/PositToBDD/PositToBDD/constraints/weighted_linear_sum_constraints2.txt", "r");
    if (file == NULL) {
        fprintf(stderr, "Could not open the file.\n");
        return 1;
    }

    printf("Constraints and BDDs:\n");

    while (fgets(line, sizeof(line), file)) {
        // Trim leading whitespace
        char *trimmed_line = line;
        while (isspace((unsigned char)*trimmed_line)) {
            trimmed_line++;
        }

        // Check if the line starts with a digit followed by a colon
        if (isdigit(trimmed_line[0])) {
            char *colon = strchr(trimmed_line, ':');
            if (colon != NULL) {
                if (constraint_count > 0) {
                    // Print a blank line to separate constraints
                    printf("\n");
                }
                constraint_count++;
                // Move pointer past the index and colon
                trimmed_line = colon + 1;
                // Trim any leading whitespace after the index and colon
                while (isspace((unsigned char)*trimmed_line)) {
                    trimmed_line++;
                }
            }
        }

        // Print the line if it is part of a constraint and not empty
        if (*trimmed_line != '\0') {
            printf("Constraint %d: %s", constraint_count, trimmed_line);
            char inputs[256] = ""; // Buffer to hold the input variables
            // Create BDD from constraint
            DdNode* bdd = createBDDFromConstraint(manager, trimmed_line, inputs);
            printf("Inputs: %s\n", inputs);
            printf("BDD created for constraint %d with %d nodes.\n", constraint_count, Cudd_DagSize(bdd));
            Cudd_RecursiveDeref(manager, bdd);
        }
    }

    fclose(file);
    printf("\nTotal number of constraints: %d\n", constraint_count);

    // Clean up CUDD
    Cudd_Quit(manager);

    return 0;
}
