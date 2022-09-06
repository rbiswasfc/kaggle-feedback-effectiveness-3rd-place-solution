#!/usr/bin/env bash

mkdir datasets
mkdir datasets/processed
mkdir models
kaggle competitions download -c feedback-prize-effectiveness
unzip feedback-prize-effectiveness.zip -d ./datasets/feedback-prize-effectiveness
kaggle competitions download -c feedback-prize-2021
unzip feedback-prize-2021.zip -d ./datasets/feedback-prize-2021
kaggle datasets download -d conjuring92/fpe-processed-dataset
unzip fpe-processed-dataset.zip -d ./datasets/processed
kaggle datasets download -d conjuring92/fpe-t5-model
unzip fpe-t5-model.zip -d ./models/T5_generator

