#!/bin/bash

# Testing script for linear probing experiments
# Usage: ./scripts/test.sh [task] [config_file] [checkpoint_path]

# Default values
TASK="cell_cls"
CONFIG_FILE="./configs/${TASK}/default.yaml"
CHECKPOINT_PATH=""
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="./results"
OUTPUT_FILE="${OUTPUT_DIR}/${TASK}_test_${TIMESTAMP}.log"
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
            echo "  config_file     Path to YAML configuration file (default: configs/[task]/default.yaml)"
            echo "  checkpoint_path Path to the trained model checkpoint (required)"
            echo ""
            echo "Examples:"
            echo "  $0 cell_cls configs/cell_cls/default.yaml checkpoints/cell_cls/ResNet50_Herlev.pth"
            echo "  $0 -b cell_cls configs/cell_cls/custom.yaml checkpoints/cell_cls/CLIP_HiCervix.pth"
            echo ""
            echo "Output will be saved to: ${OUTPUT_DIR}/[task]_test_[timestamp].log"
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
    
    # Save the process ID
    PID=$!
    echo "Process started with PID: $PID"
    echo "$PID" > "${OUTPUT_DIR}/${TASK}_test_${TIMESTAMP}.pid"
    echo "To kill the process: kill $PID"
else
    echo "Starting testing for task: $TASK with config: $CONFIG_FILE"
    echo "Using checkpoint: $CHECKPOINT_PATH"
    echo "Output will be saved to: $OUTPUT_FILE"
    
    # Run in foreground and save output
    eval "$CMD" 2>&1 | tee "$OUTPUT_FILE"
    echo "Testing completed. Output saved to: $OUTPUT_FILE"
fi