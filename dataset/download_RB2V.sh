#!/bin/bash

# Go to the dataset directory
cd dataset

# Download the datasets
wget https://huggingface.co/datasets/anhttran1111/rb2v/resolve/main/RB2Vaa
wget https://huggingface.co/datasets/anhttran1111/rb2v/resolve/main/RB2Vab
wget https://huggingface.co/datasets/anhttran1111/rb2v/resolve/main/RB2Vac

# Combine the files into a zip file
cat RB2Vaa RB2Vab RB2Vac > RB2V.zip

# Unzip the dataset
unzip RB2V.zip

# Remove the downloaded files
rm RB2Vaa RB2Vab RB2Vac RB2V.zip

# Go back to the root directory
cd ..
