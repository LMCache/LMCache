#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import shutil
import site
import importlib
from pathlib import Path
from typing import List, Tuple

def find_package_location(module_path: str) -> Tuple[str, str]:
    """
    Find the installation location for a given module path.
    Returns (base_path, module_path) tuple.
    """
    # Split the module path into parts
    parts = module_path.split('.')
    
    # Try importing the top-level package to find its location
    try:
        package = importlib.import_module(parts[0])
        base_path = os.path.dirname(package.__file__)
        return base_path, '/'.join(parts[1:])
    except ImportError:
        # If package not found, look in all site-packages
        site_packages = site.getsitepackages()
        for site_pkg in site_packages:
            potential_path = os.path.join(site_pkg, parts[0])
            if os.path.exists(potential_path):
                return site_pkg, '/'.join(parts)
        raise ImportError(f"Could not find installation location for {parts[0]}")

def read_module_list(modules_file: str) -> List[str]:
    """Read the list of modules from modules.txt"""
    with open(modules_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def copy_source_files(src_dir: str, modules: List[str]) -> None:
    """
    Copy source files to their correct locations in the Python environment.
    """
    for module in modules:
        try:
            # Find where this module should be installed
            base_path, rel_path = find_package_location(module)
            
            # Construct source and destination paths
            module_name = module.split('.')[-1]
            src_file = os.path.join(src_dir, f"{module_name}.py")
            dst_dir = os.path.join(base_path, os.path.dirname(rel_path))
            dst_file = os.path.join(dst_dir, f"{module_name}.py")

            # Create destination directory if it doesn't exist
            os.makedirs(dst_dir, exist_ok=True)

            # Copy the file
            if os.path.exists(src_file):
                print(f"Copying {src_file} -> {dst_file}")
                shutil.copy2(src_file, dst_file)
            else:
                print(f"Warning: Source file not found: {src_file}")

        except (ImportError, OSError) as e:
            print(f"Error installing {module}: {str(e)}")

def main():
    # Get the directory containing this script
    script_dir = Path(__file__).parent.absolute()
    
    # Check for vllm-v1 directory
    src_dir = os.path.join(script_dir, "vllm-v1")
    if not os.path.exists(src_dir):
        print(f"Error: Source directory not found: {src_dir}")
        sys.exit(1)

    # Read modules.txt
    modules_file = os.path.join(src_dir, "modules.txt")
    if not os.path.exists(modules_file):
        print(f"Error: modules.txt not found at {modules_file}")
        sys.exit(1)

    try:
        modules = read_module_list(modules_file)
        print(f"Found modules to install: {modules}")
        
        # Copy files to their destinations
        copy_source_files(src_dir, modules)
        print("Installation completed successfully")
        
    except Exception as e:
        print(f"Error during installation: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 