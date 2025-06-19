#!/usr/bin/env python3
"""
Test script demonstrating AI capabilities
"""

import os
import yaml
import torch
from pathlib import Path

def analyze_config(config_path):
    """Analyze a YAML configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Configuration Analysis for: {config_path}")
    print("=" * 50)
    
    # Dataset info
    if 'data' in config:
        print(f"Dataset: {config['data'].get('dataset', 'Unknown')}")
        print(f"Data Root: {config['data'].get('root', 'Unknown')}")
    
    # Model info
    if 'backbone' in config:
        print(f"Backbone: {config['backbone'].get('name', 'Unknown')}")
        print(f"Pretrained: {config['backbone'].get('pretrained', False)}")
    
    # Training info
    if 'training' in config:
        print(f"Batch Size: {config['training'].get('batch_size', 'Unknown')}")
        print(f"Epochs: {config['training'].get('epochs', 'Unknown')}")
        print(f"Learning Rate: {config['training'].get('lr', 'Unknown')}")
    
    return config

def check_gpu_availability():
    """Check GPU availability and configuration"""
    print("\nGPU Analysis:")
    print("=" * 20)
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

def main():
    """Main function demonstrating various capabilities"""
    print("AI Capability Test")
    print("=" * 50)
    
    # Check if config file exists
    config_file = "configs/cell_cls/SIPaKMeD.yaml"
    if os.path.exists(config_file):
        config = analyze_config(config_file)
        check_gpu_availability()
        
        # Suggest optimizations
        print("\nSuggested Optimizations:")
        print("=" * 30)
        if config.get('training', {}).get('batch_size', 0) > 32:
            print("- Consider reducing batch size if running out of memory")
        if config.get('training', {}).get('epochs', 0) < 10:
            print("- Consider increasing epochs for better convergence")
        print("- Monitor validation loss to prevent overfitting")
        print("- Use learning rate scheduling for better training")
    else:
        print(f"Config file not found: {config_file}")

if __name__ == "__main__":
    main()
