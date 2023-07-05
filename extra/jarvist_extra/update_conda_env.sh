#!/bin/sh

# Update and prune (install unused stuff) environment from YAML file
conda env update --name MLBD-MRes --file environment.yml --prune

# Activate this environment
conda activate MLBD-MRes

# Build python kernel (for access to dependencies e.g. pytorch)
python -m ipykernel install --user --name=MLBD-MRes

