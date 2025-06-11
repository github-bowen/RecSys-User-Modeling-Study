#!/bin/bash

# Create directories if they don't exist
mkdir -p data/raw

echo "Downloading MovieLens 1M dataset..."
wget -P data/raw/ http://files.grouplens.org/datasets/movielens/ml-1m.zip

echo "Extracting data..."
unzip data/raw/ml-1m.zip -d data/raw/
rm data/raw/ml-1m.zip

echo "MovieLens 1M dataset downloaded and extracted successfully!"
echo "Data files are located at data/raw/ml-1m/" 