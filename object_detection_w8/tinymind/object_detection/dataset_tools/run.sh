#!/bin/bash
#/media/zhou/0004DD1700005FE8/AI/00/Deep-Learning/object_detection_w8

set -e

#TRAIN_DIR=~/tmp/cifarnet-model

#DATASET_DIR=~/tmp/quiz-w8-data/images

#OUTPUT_DIR=~/tmp/quiz-w8-data/tfrecord

python object_detection/dataset_tools/create_data.py --label_map_path=/home/zhou/models/research/data/quiz-w8-data/labels_items.txt --data_dir=/home/zhou/models/research/data/quiz-w8-data --output_dir=/home/zhou/models/research/data/quiz-w8-data/tfrecord



