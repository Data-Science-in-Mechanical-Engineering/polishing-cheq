#!/bin/bash
echo "To obtain your Weights and Biases API key, visit (CTRL+click):"
echo -e "\033[4mhttps://wandb.ai/authorize\033[0m"
read -p 'wandb api key: ' wandb_api_key
#read -p 'docker image name:' image_name
#export PROJECT_DIR=$PWD
docker build -t polishing_robot \
              --build-arg WANDB_KEY=$wandb_api_key .