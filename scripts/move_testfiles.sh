#!/bin/bash

for i in {334..369}; do
    if [ -e "/mnt/ito/diffusion-anomaly/data/archive/BraTS2020_TrainingData/BraTS20_Training_${i}" ]; then
        sudo mv "/mnt/ito/diffusion-anomaly/data/archive/BraTS2020_TrainingData/BraTS20_Training_${i}" "/mnt/ito/diffusion-anomaly/data/archive/BraTS2020_ValidationData/BraTS20_Validation_${i}"
    fi
done
