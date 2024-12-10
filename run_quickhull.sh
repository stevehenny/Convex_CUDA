#!/bin/bash

# Output log file
LOG_FILE="quickhull_output.log"
LOG_FILE_OPTIMIZED="quickhull_output_optimized.log"

# Clear the log file if it exists
> "$LOG_FILE"
> "$LOG_FILE_OPTIMIZED"

# Point counts to test
POINT_COUNTS=(100 1000 10000 100000 1000000 10000000 100000000)

# Binary and command arguments
BINARY="bin/no_visual.o"
BINARY_OPTIMIZED="bin/no_visual_optimized.o"
ARGS="-c both"

# Run the algorithm for each point count and log the output
for POINTS in "${POINT_COUNTS[@]}"; do
    echo "Running for $POINTS points..." | tee -a "$LOG_FILE"
    $BINARY $ARGS -n "$POINTS" >> "$LOG_FILE" 2>&1
    echo "Completed run for $POINTS points." | tee -a "$LOG_FILE"
    echo "" >> "$LOG_FILE"
done

echo "All runs completed. Results are logged in $LOG_FILE."

for POINTS in "${POINT_COUNTS[@]}"; do
    echo "Running for $POINTS points..." | tee -a "$LOG_FILE_OPTIMIZED"
    $BINARY_OPTIMIZED $ARGS -n "$POINTS" >> "$LOG_FILE_OPTIMIZED" 2>&1
    echo "Completed run for $POINTS points." | tee -a "$LOG_FILE_OPTIMIZED"
    echo "" >> "$LOG_FILE_OPTIMIZED"
done

echo "All runs completed. Results are logged in $LOG_FILE_OPTIMIZED."
