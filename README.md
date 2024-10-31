# README
# Attention U-Net Model for Semantic Segmentation

This repository provides an Attention U-Net model implementation designed for semantic segmentation tasks, particularly optimized for UAV (Unmanned Aerial Vehicle) datasets. The model integrates attention mechanisms to enhance segmentation performance by focusing on relevant regions of the image, aiding in improved spatial accuracy and contextual understanding.

## Overview

The Attention U-Net architecture builds upon the classic U-Net structure by incorporating attention blocks. These blocks enable the model to weigh relevant spatial regions more effectively, making it ideal for applications requiring high-precision segmentation in complex or cluttered imagery.

**Note:** The UAV dataset utilized is proprietary and not publicly accessible.

## Table of Contents

- [Model Architecture](#model-architecture)
- [Attention Mechanism](#attention-mechanism)
- [Data Augmentation](#data-augmentation)
- [Training and Evaluation](#training-and-evaluation)
- [Visualization](#visualization)
- [License and Acknowledgments](#license-and-acknowledgments)

## Model Architecture

The Attention U-Net model follows the standard U-Net architecture but includes attention gates to refine the segmentation masks further:

1. **Encoder Path**: Captures features at varying levels of abstraction using convolutional layers with batch normalization and ReLU activation, followed by max pooling layers.
2. **Bottleneck Layer**: Serves as the high-level feature extractor at the center of the architecture.
3. **Decoder Path**: Gradually restores spatial dimensions through upsampling while combining skip connections with attention blocks.
4. **Attention Gates**: Applied in skip connections between encoder and decoder layers, these gates weigh relevant spatial information to improve mask precision.
5. **Output Layer**: Outputs the segmentation mask through a 1x1 convolution layer with a sigmoid activation.

```python
# Model Instantiation
model = attention_unet(input_shape, num_classes)
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
```

## Attention Mechanism

The attention mechanism in this model is based on attention gates, which work as follows:
- **Attention Block**: The attention block takes inputs from the encoder and decoder paths, computes relevance scores, and enhances critical spatial information.
- **Channel Reduction and Gating**: This mechanism allows selective focus by refining the feature maps based on relevance, thus optimizing segmentation for complex images.

The attention blocks contribute significantly to the model's effectiveness on datasets where target regions may be occluded or hard to distinguish from the background.

## Data Augmentation

Data augmentation is utilized to increase model robustness in limited-data scenarios. Techniques include:

- **Rotation**: Random rotations to simulate different orientations.
- **Shifts**: Horizontal and vertical shifts for varied object placements.
- **Zooming**: Scaling adjustments to account for size variability.
- **Horizontal Flip**: Reflecting images to increase variation.
- **Normalization**: Rescaling of pixel values to improve training stability.

This augmentation setup enables model comparisons with and without augmentation, providing insights into the effects of data enhancement.

## Training and Evaluation

The model is trained with monitoring callbacks to optimize performance:

- **ModelCheckpoint**: Saves the model with the best validation performance.
- **EarlyStopping**: Halts training if validation loss stagnates.
- **Prediction Visualization**: Visualizations are saved per epoch, aiding in visual inspection of prediction quality.

**Metrics Used**:
- **Precision**: Measures the exactness of predictions.
- **F1 Score**: Balances precision and recall.
- **Mean IoU (Intersection over Union)**: Assesses overlap accuracy.

The model’s learning progress, training loss, and validation metrics are plotted for thorough performance analysis. Evaluation also includes confusion matrices and prediction speed for real-time applications.

## Visualization

The repository includes a visualization module for generating segmentation predictions for each epoch, enabling qualitative analysis. Comparing ground truth masks with predictions offers additional insights into model strengths and areas for potential improvement.

## License and Acknowledgments

This Attention U-Net model is based on the architecture discussed in **[Attention U-Net: Learning Where to Look for the Pancreas](https://arxiv.org/abs/1804.03999)** by Oktay et al.

Attributions for any utilized libraries, such as TensorFlow, Keras, and OpenCV, are extended to their respective authors.

---

**Disclaimer**: The results presented may vary due to modifications made to the original U-Net architecture to suit UAV dataset specifications.
```