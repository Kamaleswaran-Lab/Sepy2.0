#!/usr/bin/env python3
"""
Simple script to read a YAML file in Python.
"""

import yaml
import sys
from pathlib import Path


def read_yaml_file(file_path):
    """
    Read and parse a YAML file.
    
    Args:
        file_path (str): Path to the YAML file
        
    Returns:
        dict: Parsed YAML content
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        yaml.YAMLError: If the file isn't valid YAML
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
            return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        raise
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        raise


def main():
    """Main function to demonstrate YAML reading."""
    # Example usage - modify the path as needed
    yaml_file = "configurations/emory_config.yaml"
    
    # Check if file exists
    if not Path(yaml_file).exists():
        print(f"File '{yaml_file}' not found. Please specify a valid YAML file path.")
        return
    
    try:
        # Read the YAML file
        config_data = read_yaml_file(yaml_file)
        
        # Display the contents
        print(f"Successfully read YAML file: {yaml_file}")
        print("\nContents:")
        print([[method_name] for method_name, params in config_data["yearly_instance"].items()])
        
    except Exception as e:
        print(f"Failed to read YAML file: {e}")


if __name__ == "__main__":
    main()
