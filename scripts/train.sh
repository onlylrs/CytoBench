#!/bin/bash

# Training script for linear probing experiments
# Usage: ./scripts/train.sh [task] [config_file]

# Default values
TASK="cell_cls"
CONFIG_FILE="./configs/${TASK}/base_config.yaml"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="./results/${TASK}"

# Function to extract config values
extract_config_value() {
    local config_file="$1"
    local key="$2"
    local section="$3"

    if [ -n "$section" ]; then
        # Extract from specific section (e.g., backbone.name)
        awk -v section="$section" -v key="$key" '
        BEGIN { in_section = 0 }
        /^[a-zA-Z_][a-zA-Z0-9_]*:/ {
            current_section = $1
            gsub(/:/, "", current_section)
            in_section = (current_section == section)
        }
        in_section && $1 ~ "^" key ":" {
            value = $2
            gsub(/["\x27]/, "", value)  # Remove quotes
            gsub(/[-\/]/, "_", value)   # Replace special chars
            print value
            exit
        }' "$config_file"
    else
        # Extract from any section
        grep -E "^\s*${key}:" "$config_file" | head -1 | sed "s/.*${key}:\s*[\"']*\([^\"']*\)[\"']*.*/\1/" | sed 's/[-\/]/_/g'
    fi
}

# Function to generate experiment name
generate_experiment_name() {
    # Use experiment name from Python script if available, otherwise extract from config
    if [ -n "$EXPERIMENT_NAME" ]; then
        echo "📝 Using experiment name from Python script: ${EXPERIMENT_NAME}"
    else
        # Extract backbone and dataset from config file for consistent naming
        if [ -f "$CONFIG_FILE" ]; then
            # Extract backbone name from backbone section
            BACKBONE=$(extract_config_value "$CONFIG_FILE" "name" "backbone")
            # Extract dataset name from data section
            DATASET=$(extract_config_value "$CONFIG_FILE" "dataset" "data")

            if [ -n "$BACKBONE" ] && [ -n "$DATASET" ]; then
                EXPERIMENT_NAME="${BACKBONE}_${DATASET}_${TIMESTAMP}"
                echo "📝 Using experiment name from config: ${EXPERIMENT_NAME}"
                echo "   Backbone: ${BACKBONE}"
                echo "   Dataset: ${DATASET}"
                echo "   Timestamp: ${TIMESTAMP}"
            else
                EXPERIMENT_NAME="${TASK}_${TIMESTAMP}"
                echo "⚠️  Could not extract backbone/dataset from config, using: ${EXPERIMENT_NAME}"
            fi
        else
            EXPERIMENT_NAME="${TASK}_${TIMESTAMP}"
            echo "⚠️  Config file not found, using: ${EXPERIMENT_NAME}"
        fi
    fi
}
BACKGROUND=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -b|--background)
            BACKGROUND=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options] [task] [config_file]"
            echo ""
            echo "Options:"
            echo "  -b, --background    Run in background using nohup"
            echo "  -h, --help          Show this help message"
            echo ""
            echo "Arguments:"
            echo "  task          Task type: cell_cls, cell_det, cell_seg, WSI_cls (default: cell_cls)"
            echo "  config_file   Path to YAML configuration file (default: configs/[task]/base_config.yaml)"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Train cell_cls with default config"
            echo "  $0 cell_cls                           # Train cell_cls with default config"
            echo "  $0 cell_cls configs/cell_cls/custom.yaml  # Train cell_cls with custom config"
            echo "  $0 -b cell_cls                        # Train cell_cls in background"
            echo ""
            echo "Output will be saved to: ./results/[task]/[task]_[timestamp].log"
            echo ""
            exit 0
            ;;
        *)
            # First non-option argument is the task
            if [[ -z "$TASK_ARG" ]]; then
                TASK_ARG="$1"
                TASK="$TASK_ARG"
                CONFIG_FILE="./configs/${TASK}/base_config.yaml"
                shift
            # Second non-option argument is the config file
            elif [[ -z "$CONFIG_ARG" ]]; then
                CONFIG_ARG="$1"
                CONFIG_FILE="$CONFIG_ARG"
                shift
            else
                echo "Error: Unknown argument: $1"
                exit 1
            fi
            ;;
    esac
done

# Create results directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Check if config file exists
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Configuration file not found: $CONFIG_FILE"
    exit 1
fi

# Generate experiment name after config file is confirmed to exist
generate_experiment_name

# Set output file path with experiment name
OUTPUT_FILE="${OUTPUT_DIR}/${EXPERIMENT_NAME}.log"

# Prepare the command to run
case $TASK in
    cell_cls)
        CMD="python tools/train.py --config \"$CONFIG_FILE\""
        ;;
    cell_det)
        CMD="python tools/train_det.py --config \"$CONFIG_FILE\""
        ;;
    cell_seg)
        CMD="python tools/train_seg.py --config \"$CONFIG_FILE\""
        ;;
    WSI_cls)
        echo "WSI classification training not implemented yet"
        exit 1
        ;;
    *)
        echo "Unknown task: $TASK"
        echo "Available tasks: cell_cls, cell_det, cell_seg, WSI_cls"
        exit 1
        ;;
esac

# Run the command
if [ "$BACKGROUND" = true ]; then
    echo "Starting training in background mode for task: $TASK with config: $CONFIG_FILE"
    echo "Output will be saved to: $OUTPUT_FILE"
    echo "Use 'tail -f $OUTPUT_FILE' to monitor progress"

    # Run in background with nohup
    nohup bash -c "$CMD" > "$OUTPUT_FILE" 2>&1 &

    # Show process ID for reference but don't save to file
    PID=$!
    echo "Process started with PID: $PID"
    echo "To kill the process: kill $PID"
else
    echo "Starting training for task: $TASK with config: $CONFIG_FILE"
    echo "Output will be saved to: $OUTPUT_FILE"
    
    # Run in foreground and save output
    eval "$CMD" 2>&1 | tee "$OUTPUT_FILE"
    echo "Training completed. Output saved to: $OUTPUT_FILE"
fi
