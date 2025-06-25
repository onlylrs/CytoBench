#!/bin/bash

# Testing script for linear probing experiments
# Usage: ./scripts/test.sh [task] [config_file] [checkpoint_path]

# Default values
TASK="cell_cls"
CONFIG_FILE="./configs/${TASK}/base_config.yaml"
CHECKPOINT_PATH=""
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

OUTPUT_FILE="${OUTPUT_DIR}/${EXPERIMENT_NAME}_test.log"
BACKGROUND=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -b|--background)
            BACKGROUND=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options] [task] [config_file] [checkpoint_path]"
            echo ""
            echo "Options:"
            echo "  -b, --background    Run in background using nohup"
            echo "  -h, --help          Show this help message"
            echo ""
            echo "Arguments:"
            echo "  task            Task type: cell_cls, cell_det, cell_seg, WSI_cls (default: cell_cls)"
            echo "  config_file     Path to YAML configuration file (default: configs/[task]/base_config.yaml)"
            echo "  checkpoint_path Path to the trained model checkpoint (required)"
            echo ""
            echo "Examples:"
            echo "  $0 cell_cls configs/cell_cls/default.yaml checkpoints/cell_cls/ResNet50_Herlev.pth"
            echo "  $0 -b cell_cls configs/cell_cls/custom.yaml checkpoints/cell_cls/CLIP_HiCervix.pth"
            echo ""
            echo "Output will be saved to: ./results/[task]/[task]_test_[timestamp].log"
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
            # Third non-option argument is the checkpoint path
            elif [[ -z "$CHECKPOINT_ARG" ]]; then
                CHECKPOINT_ARG="$1"
                CHECKPOINT_PATH="$CHECKPOINT_ARG"
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

# Check if checkpoint path is provided
if [[ -z "$CHECKPOINT_PATH" ]]; then
    echo "Error: Checkpoint path is required"
    echo "Usage: $0 [options] [task] [config_file] [checkpoint_path]"
    exit 1
fi

# Check if checkpoint file exists
if [[ ! -f "$CHECKPOINT_PATH" ]]; then
    echo "Error: Checkpoint file not found: $CHECKPOINT_PATH"
    exit 1
fi

# Update output file path with task name
OUTPUT_FILE="${OUTPUT_DIR}/${TASK}_test_${TIMESTAMP}.log"

# Prepare the command to run
case $TASK in
    cell_cls)
        CMD="python tools/test.py --config \"$CONFIG_FILE\" --checkpoint \"$CHECKPOINT_PATH\""
        ;;
    cell_det)
        CMD="python tools/test_det.py --config \"$CONFIG_FILE\" --checkpoint \"$CHECKPOINT_PATH\""
        ;;
    cell_seg)
        CMD="python tools/test_seg.py --config \"$CONFIG_FILE\" --checkpoint \"$CHECKPOINT_PATH\""
        ;;
    WSI_cls)
        echo "WSI classification testing not implemented yet"
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
    echo "Starting testing in background mode for task: $TASK with config: $CONFIG_FILE"
    echo "Using checkpoint: $CHECKPOINT_PATH"
    echo "Output will be saved to: $OUTPUT_FILE"
    echo "Use 'tail -f $OUTPUT_FILE' to monitor progress"

    # Run in background with nohup
    nohup bash -c "$CMD" > "$OUTPUT_FILE" 2>&1 &

    # Show process ID for reference but don't save to file
    PID=$!
    echo "Process started with PID: $PID"
    echo "To kill the process: kill $PID"
else
    echo "Starting testing for task: $TASK with config: $CONFIG_FILE"
    echo "Using checkpoint: $CHECKPOINT_PATH"
    echo "Output will be saved to: $OUTPUT_FILE"
    
    # Run in foreground and save output
    eval "$CMD" 2>&1 | tee "$OUTPUT_FILE"
    echo "Testing completed. Output saved to: $OUTPUT_FILE"
fi