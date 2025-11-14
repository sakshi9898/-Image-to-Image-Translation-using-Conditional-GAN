# Pix2Pix Image-to-Image Translation

## Overview
This project implements a Pix2Pix generative adversarial network (GAN) for image-to-image translation. It uses the Facades dataset and trains a U-Net-based generator with a PatchGAN discriminator.

## Features
- Automatic dataset download and extraction.
- Preprocessing pipeline to split and normalize paired images.
- U-Net generator and PatchGAN discriminator.
- Custom training step using TensorFlow.
- Displays input, ground truth, and generated output.

## Dataset
The Facades dataset is downloaded from:
http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.tar.gz

The dataset contains paired images: left side is the real facade, right side is the sketch.

## How It Works
1. Dataset is downloaded and extracted.
2. Each image is split into input and target halves.
3. Generator converts input image to target style.
4. Discriminator classifies real vs generated pairs.
5. A simple training loop demonstrates model training.

## Model Architecture
- Generator: U-Net with skip connections.
- Discriminator: PatchGAN classifier.

## How to Run
Run the script:

python pix2pix.py

Use Google Colab or a machine with GPU for best performance.

## Output
The script displays:
- Input Image
- Ground Truth
- Generated Image

The model demonstrates image-to-image translation after training.
