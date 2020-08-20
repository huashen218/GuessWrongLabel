#!/usr/bin/env bash
set -x;
set -e;

export PYTHONPATH=/your_folder_dir/

MODEL="resnet50";
DATASET="imagenet";

IMAGENET_DATADIR="/your_dataset_dir";
INCORRECT_IMGS_DIR="/your_incorrect_image_dir/incorrect_resnet50_imagenet.npz"; 
HEATMAPS_DIR="/your_heatmap_dir/incorrect_heatmaps"; 

CUDA_VISIBLE_DEVICES=0 python imagenet_generate_heatmaps.py --incorrect_img_dir $INCORRECT_IMGS_DIR --save_dir $HEATMAPS_DIR -d $DATASET -m $MODEL -r 2

