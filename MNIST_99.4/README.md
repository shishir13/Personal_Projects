# MNIST Classification with PyTorch

This project implements a CNN model for MNIST digit classification with the following specifications:
- Achieves 99.4% validation accuracy
- Uses less than 20k parameters
- Trains in less than 20 epochs
- Implements Batch Normalization and Dropout
- Uses Fully Connected layers

## Model Architecture
- Input Layer
- Convolutional layers with Batch Normalization
- Max Pooling layers
- Dropout for regularization
- Fully Connected layers
- Output layer (10 classes)

## Requirements
- PyTorch
- torchvision
- Python 3.8+

## Project Structure
- `models/model.py`: Model architecture definition
- `train.py`: Training script
- `utils.py`: Utility functions
- `tests/`: Test cases
- `.github/workflows/`: CI/CD pipeline

## Usage
```bash
# Train the model
python train.py

# Run tests
python -m pytest tests/test_model.py
```
