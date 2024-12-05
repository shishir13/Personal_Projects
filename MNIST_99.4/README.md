# MNIST 99.4% Accuracy Challenge

This project achieves 99.43% accuracy on MNIST digit classification while maintaining less than 20,000 parameters.

## Model Architecture
- Input: 1 channel image (28x28)
- Convolutional layers:
  * Conv1: 1 -> 8 channels
  * Conv2: 8 -> 16 channels
  * Conv3: 16 -> 32 channels
- Each conv block includes:
  * 3x3 convolution with padding=1
  * BatchNorm
  * ReLU activation
  * MaxPool2d
- Fully connected layers:
  * FC1: 32 * 3 * 3 -> 32
  * FC2: 32 -> 10
- Dropout (0.3) after conv3 and FC1

Total Parameters: 15,578

## Training Configuration
- Epochs: 19
- Batch size: 128
- Optimizer: Adam
  * Learning rate: 0.001
  * Weight decay: 1e-4
- Scheduler: OneCycleLR
  * Max learning rate: 0.003
  * pct_start: 0.2
  * div_factor: 10
  * final_div_factor: 100
- Loss: CrossEntropyLoss

## Data Augmentation
- Random rotation (±10 degrees)
- Random translation (±10%)
- Normalization (mean=0.1307, std=0.3081)

## Results
- Best Test Accuracy: 99.43%
- Parameters: 15,578 (under 20k limit)
- Training Time: 19 epochs

## Requirements
- Python 3.8+
- PyTorch
- torchvision
- numpy
- matplotlib
- tqdm

## Usage
```bash
python train.py
```

## Project Structure
```
MNIST_99.4/
├── models/
│   └── model.py      # Model architecture
├── train.py          # Training script
└── README.md         # Documentation
