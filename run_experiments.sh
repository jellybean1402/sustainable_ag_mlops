#!/bin/bash

# This script runs a grid search of experiments for the RandomForest model.

echo "--- Starting DVC Experiment Grid Search ---"

# Define the hyperparameters to test
N_ESTIMATORS=(50 100 150 200)
MAX_DEPTH=(10 20 30 50)

# --- CHANGE 1: Queue experiments instead of running them directly ---
# Loop through each combination and add it to the experiment queue.
for estimators in "${N_ESTIMATORS[@]}"; do
  for depth in "${MAX_DEPTH[@]}"; do
    EXP_NAME="exp-${estimators}-${depth}"
    echo "--- Queuing Experiment: ${EXP_NAME} ---"
    
    # The --queue flag is the crucial change. It stages the experiment without running it.
    dvc exp run \
      --queue \
      --set-param "train.n_estimators=${estimators}" \
      --set-param "train.max_depth=${depth}" \
      -n "${EXP_NAME}"
  done
done

echo "--- All experiments have been queued. Now running them all. ---"

# --- CHANGE 2: Run all the queued experiments ---
# The --run-all flag executes every experiment in the queue.
dvc exp run --run-all

echo "--- All experiments have been run. ---"
