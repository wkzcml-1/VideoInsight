#!/bin/bash

# conda environment name
ENV_NAME="vi"

# setup mongodb, milvus by docker\
# create database directory
if [ ! -d "database" ]; then
    mkdir database
fi
# create database/mongodb_data database/milvus_data directory
if [ ! -d "database/mongodb_data" ]; then
    mkdir database/mongodb_data
fi
if [ ! -d "database/milvus_data" ]; then
    mkdir database/milvus_data
fi

# start mongodb, milvus by docker
# if mongodb, milvus is not running, restart it unless it is running
docker compose up -d

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
        CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
    fi
else
    echo "requirements.txt not found. Skipping dependency installation."
    exit 1
fi

echo "Environment setup complete."

# download openbmb/MiniCPM-V-2_6 model
# check if checkpoint directory exists
MiniCPM_MODEL="openbmb/MiniCPM-V-2_6-gguf"
MiniCPM_GGUF_VERSION="ggml-model-Q8_0.gguf"
MiniCPM_DIR="./checkpoints/$MiniCPM_MODEL"
huggingface-cli download  $MiniCPM_MODEL $MiniCPM_GGUF_VERSION --local-dir $MiniCPM_DIR

# download BAAI/bge-m3 model
BGE_M3_MODEL="BAAI/bge-m3"
BGE_M3_DIR="./checkpoints/$BGE_M3_MODEL"
huggingface-cli download  $BGE_M3_MODEL --local-dir $BGE_M3_DIR

# download BAAI/bge-reranker-v2-m3 model
BGE_RERANKER_V2_M3_MODEL="BAAI/bge-reranker-v2-m3"
BGE_RERANKER_V2_M3_DIR="./checkpoints/$BGE_RERANKER_V2_M3_MODEL"
huggingface-cli download  $BGE_RERANKER_V2_M3_MODEL --local-dir $BGE_RERANKER_V2_M3_DIR

# download BAAI/AltCLIP model
ALTCLIP_MODEL="BAAI/AltCLIP"
ALTCLIP_DIR="./checkpoints/$ALTCLIP_MODEL"
huggingface-cli download  $ALTCLIP_MODEL --local-dir $ALTCLIP_DIR
 
# download Systran/faster-distil-whisper-large-v3
WHISPER_MODEL="openai/whisper-large-v3"
WHISPER_DIR="./checkpoints/$WHISPER_MODEL"
huggingface-cli download  $WHISPER_MODEL --local-dir $WHISPER_DIR

# if not exist dataset directory, create it
if [ ! -d "dataset" ]; then
    mkdir dataset
fi

# download dataset
