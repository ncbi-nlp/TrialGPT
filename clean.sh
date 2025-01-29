#!/bin/bash

# Remove the 'results' directory and its contents, verbosely
rm -rfv results

# Create a new 'results' directory
mkdir results

# Copy __init__.py from results_cuwtf to results, verbosely
touch results/__init__.py