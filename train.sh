#!/bin/bash

# Detect OS (Linux, macOS, Windows)
OS_TYPE="Unknown"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS_TYPE="Linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS_TYPE="macOS"
elif [[ "$OSTYPE" == "cygwin" || "$OSTYPE" == "msys" ]]; then
    OS_TYPE="Windows"
else
    OS_TYPE="Unsupported OS"
fi

echo "Detected OS: $OS_TYPE"

# Default values for training arguments
RANDOM_STATE=42
N_ESTIMATORS=100
SPLIT_SIZE=0.2

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --random_state) RANDOM_STATE="$2"; shift ;;
        --n_estimators) N_ESTIMATORS="$2"; shift ;;
        --split_size) SPLIT_SIZE="$2"; shift ;;
        *) echo "Unknown parameter: $1" ;;
    esac
    shift
done

# Run the training pipeline with the specified arguments
echo "Starting Training with random_state=$RANDOM_STATE, n_estimators=$N_ESTIMATORS, split_size=$SPLIT_SIZE"
python model_train.py --random_state $RANDOM_STATE --n_estimators $N_ESTIMATORS --split_size $SPLIT_SIZE

# Exit script
exit 0
