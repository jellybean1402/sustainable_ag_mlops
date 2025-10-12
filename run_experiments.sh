#!/bin/bash

# This script runs a grid search, captures the metric from each run,
# and identifies the best-performing experiment.

set -e # Exit immediately if a command exits with a non-zero status.

echo "--- Starting DVC Experiment Grid Search ---"

# Initialize variables to track the best experiment
BEST_EXP_NAME=""
# Initialize with a very low number so the first score is always higher
BEST_SCORE=-1.0

# Define the hyperparameters to test
N_ESTIMATORS=(50 100 150 200)
MAX_DEPTH=(5 10 15 20) # Removed 'none' for simplicity in this robust script

# Loop through each combination of hyperparameters
for estimators in "${N_ESTIMATORS[@]}"; do
  for depth in "${MAX_DEPTH[@]}"; do
    EXP_NAME="exp-${estimators}-${depth}"
    echo "--- Running Experiment: ${EXP_NAME} ---"
    
    # Run a single experiment and capture its JSON output
    # This is a robust way to get the results directly.
    RESULT_JSON=$(dvc exp run \
      --set-param "train.n_estimators=${estimators}" \
      --set-param "train.max_depth=${depth}" \
      -n "${EXP_NAME}" \
      --json)
      
    # Use jq to parse the r2_score from the result JSON
    CURRENT_SCORE=$(echo $RESULT_JSON | jq -r '.[].metrics."reports/metrics.json".data.r2_score')
    
    echo "Experiment ${EXP_NAME} finished with R^2 score: ${CURRENT_SCORE}"
    
    # Compare the current score with the best score found so far
    # The 'bc' command is used for floating-point number comparison
    if (( $(echo "$CURRENT_SCORE > $BEST_SCORE" | bc -l) )); then
      echo "New best score found!"
      BEST_SCORE=$CURRENT_SCORE
      BEST_EXP_NAME=$EXP_NAME
    fi
  done
done

echo "--- Grid Search Complete ---"
echo "Best experiment found: ${BEST_EXP_NAME}"
echo "Best R^2 score: ${BEST_SCORE}"

# Crucial final step: Write the name of the best experiment to a file
# that the GitHub Actions workflow can read.
echo "${BEST_EXP_NAME}" > best_experiment.txt
