#!/bin/bash

# Exit immediately if a command exits with a non-zero status
# set -e
# how to run with log file below command
# clear && source setup_trialgpt_env.sh |& tee log_build_trialgpt_env.txt

# Name of the new Conda environment
ENV_NAME="trialgpt_env"

# Check if the environment exists and remove it
if conda env list | grep -q "^$ENV_NAME "; then
    echo "Environment $ENV_NAME exists. Removing it..."
    conda deactivate
    conda env remove -n $ENV_NAME -y
fi

# Create a new Conda environment with Python 3.12
echo "Creating new environment: $ENV_NAME"
conda create -n $ENV_NAME python=3.12 -y

# Activate the new environment
echo "Activating environment: $ENV_NAME"
source activate $ENV_NAME

# Function to clone repository if it doesn't exist
clone_if_not_exists() {
    if [ ! -d "$1" ]; then
        echo "Cloning repository: $1"
        git clone "https://github.com/$1.git"
    else
        echo "Repository $1 already exists. Skipping clone."
        echo "If you want to update the repository, consider running 'git pull' in the $1 directory."
    fi
}

# Clone TrialGPT repository if it doesn't exist
#clone_if_not_exists "ncbi-nlp/TrialGPT"

# Install build essentials and common data science packages
echo "Installing build essentials and common packages..."
conda install -y gcc_linux-64 gxx_linux-64 numpy==1.26.4 pandas==2.2.2 pyspark==3.5.3 scikit-learn==1.5.2 jupyterlab matplotlib scipy seaborn plotly ipywidgets nbconvert pyarrow

# Install PyTorch with CUDA support, torchvision, torchaudio, and faiss-gpu
echo "Installing PyTorch ecosystem..."
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia -y
conda install -c pytorch -c nvidia faiss-gpu=1.9.0 -y
echo "PyTorch installation complete"

# Install additional required packages
echo "Installing additional packages..."
pip install transformers==4.48.1 nltk==3.8.1 openai==1.59.7 rank_bm25==0.2.2 accelerate bitsandbytes tqdm==4.67.1

echo ""
echo "Environment $ENV_NAME has been created and packages have been installed."
echo "To activate the environment, use: conda activate $ENV_NAME"
echo "Remember to set up your OpenAI API environment variables:"
echo 'export OPENAI_API_KEY="sk-proj-"'
echo ""
echo "To start JupyterLab, activate the environment and run:"
echo "jupyter lab"

# The following block is commented out as the conda installs above cover all current requirements including PyTorch GPU support
: <<'END_COMMENT'
# TrialGPT specific setup (not needed with current conda installs)
# TRIALGPT_DIR="./TrialGPT"
# cd $TRIALGPT_DIR || { echo "Failed to change to TrialGPT directory"; exit 1; }
# pip install -r requirements.txt
END_COMMENT