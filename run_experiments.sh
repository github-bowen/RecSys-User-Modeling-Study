#!/bin/bash

# Make script executable
USER_MODELS=("demographic" "behavioral" "hybrid")
ALGORITHMS=("knn")

echo "Starting experiments for user modeling strategies comparison..."

for user_model in "${USER_MODELS[@]}"; do
  for algo in "${ALGORITHMS[@]}"; do
    echo "==========================================================="
    echo "Running experiment with User Model: $user_model and Algorithm: $algo"
    echo "-----------------------------------------------------------"
    python main.py --user_model "$user_model" --algorithm "$algo"
    echo "==========================================================="
  done
done

echo "All experiments completed successfully!"
echo "Results are saved in the results directory." 