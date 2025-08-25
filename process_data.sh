#!/bin/bash

# keel datasets
KEEL_DATASETS=("autos" "balance" "car" "cleveland" "dermatology" "ecoli" "flare" "glass" "hayes-roth" "lymphography" "newthyroid" "shuttle" "thyroid" "zoo")
UCI_DATASETS=("hcv")
LIBSVM_DATASETS=("segment" "vehicle" "aloi")

SOURCE_DIR="raw_data"
# folder to save svm data format
SAVE_SVM_DATA_DIR="svm_data"

# process libsvm
python3 data_source_processors/process_libsvm.py --datasets "${LIBSVM_DATASETS[@]}" --source_folder "$SOURCE_DIR" --save_folder "$SAVE_SVM_DATA_DIR"

