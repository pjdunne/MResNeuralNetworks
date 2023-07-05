#!/bin/sh

# Create environment from YAML file
conda env create --file environment.yml

# Activate this environment
conda activate MLBD-MRes

# Build python kernel (for access to dependencies e.g. pytorch)
python -m ipykernel install --user --name=MLBD-MRes

