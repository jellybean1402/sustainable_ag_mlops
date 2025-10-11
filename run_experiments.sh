#!/bin/bash

# This script runs a grid search of experiments for the RandomForest model.

echo "--- Starting DVC Experiment Grid Search ---"

# Define the hyperparameters to test
N_ESTIMATORS=(50 100 150 200)
MAX_DEPTH=(5 10 15 20)

# Loop through each combination of hyperparameters
for estimators in "${N_ESTIMATORS[@]}"; do
  for depth in "${MAX_DEPTH[@]}"; do
    # Create a unique experiment name
    EXP_NAME="exp-${estimators}-${depth}"
    
    echo "--- Running Experiment: ${EXP_NAME} ---"
    
    # Run the DVC experiment
    # The --queue flag adds the experiment to a queue to be run.
    # The -n flag gives it a custom name.
    dvc exp run \
      --set-param "train.n_estimators=${estimators}" \
      --set-param "train.max_depth=${depth}" \
      -n "${EXP_NAME}"
  done
done

echo "--- All experiments have been queued and run. ---"
