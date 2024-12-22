#!/bin/bash

# Script to clone a Git repository

# Function to display usage information
usage() {
    echo "Usage: $0 [destination-directory]"
    echo "  [destination-directory] Optional: The directory to clone the repository into."
    exit 1
}

# Set repository URL
REPO_URL="https://github.com/ElShiny/AMS_izziv"

# Assign destination directory if provided
DEST_DIR=$1

# Clone the repository
if [ -z "$DEST_DIR" ]; then
    git clone "$REPO_URL"
else
    git clone "$REPO_URL" "$DEST_DIR"
fi

# Check if the clone was successful
if [ $? -eq 0 ]; then
    echo "Repository cloned successfully."

    # Ask the user if they want to download pictures
    read -p "Do you want to download pictures from the repository? (yes/no): " DOWNLOAD_PICS
    if [[ "$DOWNLOAD_PICS" == "yes" || "$DOWNLOAD_PICS" == "y" ]]; then
        mkdir -p data
        cd data || exit

        wget https://cloud.imi.uni-luebeck.de/s/xQPEy4sDDnHsmNg/download/ThoraxCBCT_OncoRegRelease_06_12_23.zip
        unzip -q -o ThoraxCBCT_OncoRegRelease_06_12_23.zip
        rm -r __MACOSX/ && rm ThoraxCBCT_OncoRegRelease_06_12_23.zip
        cd ..

        echo "Folder 'data' created and switched to it."
        echo "You can now proceed to handle picture downloads into the 'data' folder."
    else
        echo "Skipping picture download."
    fi

    # Ask the user if they want to use a W&B key
    read -p "Do you want to configure a W&B (Weights & Biases) key? (yes/no): " CONFIG_WANDB
    if [[ "$CONFIG_WANDB" == "yes" || "$CONFIG_WANDB" == "y" ]]; then
        read -p "Please enter your W&B key: " WANDB_KEY
        export WANDB_API_KEY=$WANDB_KEY
        echo "W&B key configured successfully."
    else
        echo "Skipping W&B key configuration."
    fi

    # Change directory into the repository
    if [ -z "$DEST_DIR" ]; then
        REPO_NAME=$(basename "$REPO_URL" .git)
        cd "$REPO_NAME" || exit
    else
        cd "$DEST_DIR" || exit
    fi
    echo "Changed directory to $(pwd)."
    
    # Build a Docker image from the Dockerfile
    if [ -f Dockerfile ]; then
        echo "Dockerfile found. Building Docker image..."
        docker build --build-arg KEY="$WANDB_KEY" -t repo_image .
        if [ $? -eq 0 ]; then
            echo "Docker image built successfully as 'repo_image'."
        else
            echo "Error: Failed to build Docker image."
        fi
    else
        echo "No Dockerfile found in the repository. Skipping Docker image build."
    fi
else
    echo "Error: Failed to clone repository."
    exit 2
fi