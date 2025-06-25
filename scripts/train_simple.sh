#!/bin/bash

# Simple training script with parameter override support
# Usage: ./scripts/train_simple.sh [OPTIONS] <task> <config_file> [OVERRIDES...]

# Default values
TASK="cell_cls"
CONFIG_FILE="./configs/${TASK}/base_config.yaml"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="./results/${TASK}"
BACKGROUND=false

# Function to extract config values (handles YAML inheritance)
extract_config_value() {
    local config_file="$1"
    local key="$2"
    local section="$3"

    # First try to extract from the current file
    local value=""

    if [ -n "$section" ]; then
        # Extract from specific section (e.g., backbone.name)
        value=$(awk -v section="$section" -v key="$key" '
        BEGIN { in_section = 0 }
        /^[a-zA-Z_][a-zA-Z0-9_]*:/ {
            current_section = $1
            gsub(/:/, "", current_section)
            in_section = (current_section == section)
        }
        in_section && $1 ~ "^" key ":" {
            val = $2
            gsub(/["\x27]/, "", val)  # Remove quotes
            gsub(/[-\/]/, "_", val)   # Replace special chars
            print val
            exit
        }' "$config_file")
    else
        # Extract from any section
        value=$(grep -E "^\s*${key}:" "$config_file" | head -1 | sed "s/.*${key}:\s*[\"']*\([^\"']*\)[\"']*.*/\1/" | sed 's/[-\/]/_/g')
    fi

    # If not found and there's a _base_ reference, try the base file
    if [ -z "$value" ]; then
        local base_file=$(grep "^_base_:" "$config_file" | sed 's/_base_:\s*["\x27]*\([^"\x27]*\)["\x27]*.*/\1/')
        if [ -n "$base_file" ]; then
            # Resolve relative path
            local config_dir=$(dirname "$config_file")
            local base_path="${config_dir}/${base_file}"
            if [ -f "$base_path" ]; then
                value=$(extract_config_value "$base_path" "$key" "$section")
            fi
        fi
    fi

    echo "$value"
}

show_help() {
    echo "Usage: $0 [OPTIONS] <task> <config_file> [OVERRIDES...]"
    echo ""
    echo "Train a model for the specified task using the given configuration."
    echo ""
    echo "Arguments:"
    echo "  task          Task type (e.g., cell_cls, cell_det, cell_seg)"
    echo "  config_file   Path to YAML configuration file"
    echo "  OVERRIDES     Configuration overrides in format key=value"
    echo ""
    echo "Options:"
    echo "  -h, --help    Show this help message and exit"
    echo "  -b, --batch   Run in batch mode (redirect output to log file)"
    echo ""
    echo "Configuration Overrides:"
    echo "  Use dot notation to override nested config values:"
    echo "    common.gpu=1              Set GPU device"
    echo "    data.dataset=SIPaKMeD     Set dataset"
    echo "    backbone.name=ResNet18    Set backbone model"
    echo ""
    echo "Examples:"
    echo "  $0 cell_cls configs/cell_cls/base_config.yaml"
    echo "  $0 -b cell_cls configs/cell_cls/quick_test.yaml common.gpu=1 data.dataset=SIPaKMeD"
    echo ""
}

# Parse arguments
OVERRIDE_ARGS=""
while [ $# -gt 0 ]; do
    case "$1" in
        -b|--batch)
            BACKGROUND=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            # First non-option argument is the task
            if [ -z "$TASK_ARG" ]; then
                TASK_ARG="$1"
                TASK="$TASK_ARG"
                CONFIG_FILE="./configs/${TASK}/base_config.yaml"
                OUTPUT_DIR="./results/${TASK}"
                shift
            # Second non-option argument is the config file
            elif [ -z "$CONFIG_ARG" ]; then
                CONFIG_ARG="$1"
                CONFIG_FILE="$CONFIG_ARG"
                shift
            # Remaining arguments are overrides
            else
                # Check if argument contains '=' (override format)
                if echo "$1" | grep -q "="; then
                    OVERRIDE_ARGS="$OVERRIDE_ARGS --override $1"
                    echo "🔧 Override: $1"
                    shift
                else
                    echo "Error: Invalid argument format: $1"
                    echo "Overrides should be in format key=value"
                    exit 1
                fi
            fi
            ;;
    esac
done

# Create results directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found: $CONFIG_FILE"
    exit 1
fi

# Function to generate experiment name (must be defined before use)
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

            # Debug output
            echo "🔍 Debug: Extracted from config:"
            echo "   BACKBONE: '$BACKBONE'"
            echo "   DATASET: '$DATASET'"

            # Apply overrides if they exist
            if [ -n "$OVERRIDE_ARGS" ]; then
                # Extract backbone and dataset from override args
                BACKBONE_OVERRIDE=$(echo "$OVERRIDE_ARGS" | grep -o "backbone\.name=[^ ]*" | cut -d'=' -f2 | head -1)
                DATASET_OVERRIDE=$(echo "$OVERRIDE_ARGS" | grep -o "data\.dataset=[^ ]*" | cut -d'=' -f2 | head -1)

                if [ -n "$BACKBONE_OVERRIDE" ]; then
                    BACKBONE=$(echo "$BACKBONE_OVERRIDE" | sed 's/[-\/]/_/g')
                    echo "🔧 Override: BACKBONE set to '$BACKBONE'"
                fi

                if [ -n "$DATASET_OVERRIDE" ]; then
                    DATASET="$DATASET_OVERRIDE"
                    echo "🔧 Override: DATASET set to '$DATASET'"
                fi
            fi

            # Clean up backbone name (replace special characters)
            if [ -n "$BACKBONE" ]; then
                BACKBONE=$(echo "$BACKBONE" | sed 's/[-\/]/_/g')
            fi

            if [ -n "$BACKBONE" ] && [ -n "$DATASET" ]; then
                EXPERIMENT_NAME="${BACKBONE}_${DATASET}_${TIMESTAMP}"
                echo "📝 Using experiment name: ${EXPERIMENT_NAME}"
                echo "   Backbone: ${BACKBONE}"
                echo "   Dataset: ${DATASET}"
                echo "   Timestamp: ${TIMESTAMP}"
            else
                EXPERIMENT_NAME="${TASK}_${TIMESTAMP}"
                echo "⚠️  Could not extract backbone/dataset from config, using: ${EXPERIMENT_NAME}"
                echo "   Missing: BACKBONE='$BACKBONE', DATASET='$DATASET'"
            fi
        else
            EXPERIMENT_NAME="${TASK}_${TIMESTAMP}"
            echo "⚠️  Config file not found, using: ${EXPERIMENT_NAME}"
        fi
    fi
}

# Generate experiment name after config file is confirmed to exist
generate_experiment_name

# Set output file path with experiment name
OUTPUT_FILE="${OUTPUT_DIR}/${EXPERIMENT_NAME}.log"

# Prepare the command to run
case $TASK in
    cell_cls)
        CMD="python tools/train.py --config \"$CONFIG_FILE\"$OVERRIDE_ARGS"
        ;;
    cell_det)
        CMD="python tools/train_det.py --config \"$CONFIG_FILE\"$OVERRIDE_ARGS"
        ;;
    cell_seg)
        CMD="python tools/train_seg.py --config \"$CONFIG_FILE\"$OVERRIDE_ARGS"
        ;;
    *)
        echo "Unknown task: $TASK"
        echo "Available tasks: cell_cls, cell_det, cell_seg"
        exit 1
        ;;
esac

# Run the command
if [ "$BACKGROUND" = true ]; then
    echo "Starting training in background mode for task: $TASK with config: $CONFIG_FILE"
    echo "Output will be saved to: $OUTPUT_FILE"
    echo "Command: $CMD"
    echo "Use 'tail -f $OUTPUT_FILE' to monitor progress"

    # Run in background with nohup
    nohup bash -c "$CMD" > "$OUTPUT_FILE" 2>&1 &

    # Show process ID for reference
    PID=$!
    echo "Process started with PID: $PID"
    echo "To kill the process: kill $PID"
else
    echo "Starting training for task: $TASK with config: $CONFIG_FILE"
    echo "Command: $CMD"
    echo "Output will be saved to: $OUTPUT_FILE"
    
    # Run in foreground and save output
    eval "$CMD" 2>&1 | tee "$OUTPUT_FILE"
    echo "Training completed. Output saved to: $OUTPUT_FILE"
fi
