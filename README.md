Root Directory

README.md: Instructions for non-technical users on how to use the repository.
requirements.txt: List of dependencies to install.
.gitignore: Standard Git ignore file.
/src (Source Code Directory)

main.py: The main script for users to run the whole process.
/data_processing
image_loader.py: Functions to load and handle images.
tiff_processor.py: Functions specific to TIFF file processing.
/model
model_handler.py: Functions related to loading and using the machine learning models.
/utils
visualization.py: Functions for data visualization.
file_utils.py: Utility functions for file operations.
config_utils.py: Handling of configuration settings.
/data

Directory for storing input images and other data files.
/output

Directory where processed images and results are saved.
/config

config.json: Configuration file for users to specify parameters.
/notebooks (optional)

For any future Jupyter notebooks, if needed for demonstration or testing.

"""
# Conda Environment Setup for Your Project

This guide will help you create and use a Conda environment with the required packages for your project. The necessary packages and their versions are listed in the `requirements.txt` file.

## Prerequisites

- [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on your system.

## Setup Instructions
cd /mnt/c/Users/mason/Desktop/Brown/Glacial-Maps

```bash
# Create a New Conda Environment
conda create -n <environment_name> python=3.10.12

# Activate the Conda Environment
conda activate <environment_name>

# Install Packages from `requirements.txt`
conda install --file requirements.txt

# Verify Installation
conda list

