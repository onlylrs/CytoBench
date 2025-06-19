#!/bin/bash

# Training script for linear probing experiments
# Usage: ./scripts/train.sh [task] [config_file]

# Default values
TASK="cell_cls"
CONFIG_FILE="./configs/${TASK}/default.yaml"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="./results"
OUTPUT_FILE="${OUTPUT_DIR}/${TASK}_${TIMESTAMP}.log"
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
            echo "  config_file   Path to YAML configuration file (default: configs/[task]/default.yaml)"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Train cell_cls with default config"
            echo "  $0 cell_cls                           # Train cell_cls with default config"
            echo "  $0 cell_cls configs/cell_cls/custom.yaml  # Train cell_cls with custom config"
            echo "  $0 -b cell_cls                        # Train cell_cls in background"
            echo ""
            echo "Output will be saved to: ${OUTPUT_DIR}/[task]_[timestamp].log"
            echo ""
            exit 0
            ;;
        *)
            # First non-option argument is the task
            if [[ -z "$TASK_ARG" ]]; then
                TASK_ARG="$1"
                TASK="$TASK_ARG"
                CONFIG_FILE="./configs/${TASK}/default.yaml"
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

# Update output file path with task name
OUTPUT_FILE="${OUTPUT_DIR}/${TASK}_${TIMESTAMP}.log"

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
    
    # Save the process ID
    PID=$!
    echo "Process started with PID: $PID"
    echo "$PID" > "${OUTPUT_DIR}/${TASK}_${TIMESTAMP}.pid"
    echo "To kill the process: kill $PID"
else
    echo "Starting training for task: $TASK with config: $CONFIG_FILE"
    echo "Output will be saved to: $OUTPUT_FILE"
    
    # Run in foreground and save output
    eval "$CMD" 2>&1 | tee "$OUTPUT_FILE"
    echo "Training completed. Output saved to: $OUTPUT_FILE"
fi
