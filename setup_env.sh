#!/bin/bash

# conda environment name
ENV_NAME="vi"

# create conda environment if it doesn't exist
if ! conda env list | awk '{print $1}' | grep -wq "$ENV_NAME"; then
    echo "Creating Conda environment..."
    conda create --name $ENV_NAME python=3.10.4 --yes
else
    echo "Conda environment already exists."
fi

# activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# install dependencies
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies..."
    # update pip
    pip install --upgrade pip
    pip install -r requirements.txt
    # install flash-attn 
    pip install flash-attn --no-build-isolation
    # install llama.cpp with its python bindings
    # if llama-cpp-python is not installed, install it with CUDA support
    if ! pip show llama-cpp-python; then
        CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python -vv
    fi
else
    echo "requirements.txt not found. Skipping dependency installation."
fi

echo "Environment setup complete."

# download openbmb/MiniCPM-V-2_6 model
# check if checkpoint directory exists
MiniCPM_MODEL="openbmb/MiniCPM-V-2_6-gguf"
MiniCPM_GGUF_VERSION="ggml-model-Q8_0.gguf"
MiniCPM_DIR="./checkpoints/$MiniCPM_MODEL"
huggingface-cli download --resume-download $MiniCPM_MODEL $MiniCPM_GGUF_VERSION --local-dir $MiniCPM_DIR

# download BAAI/bge-m3 model
BGE_M3_MODEL="BAAI/bge-m3"
BGE_M3_DIR="./checkpoints/$BGE_M3_MODEL"
huggingface-cli download --resume-download $BGE_M3_MODEL --local-dir $BGE_M3_DIR

# download laion/larger_clap_general
LARGER_CLAP_MODEL="laion/larger_clap_general"
LARGER_CLAP_DIR="./checkpoints/$LARGER_CLAP_MODEL"
huggingface-cli download --resume-download $LARGER_CLAP_MODEL --local-dir $LARGER_CLAP_DIR

# download Systran/faster-distil-whisper-large-v3
WHISPER_MODEL="Systran/faster-distil-whisper-large-v3"
WHISPER_DIR="./checkpoints/$WHISPER_MODEL"
huggingface-cli download --resume-download $WHISPER_MODEL --local-dir $WHISPER_DIR

# if not exist dataset directory, create it
if [ ! -d "dataset" ]; then
    mkdir dataset
fi

# download dataset
