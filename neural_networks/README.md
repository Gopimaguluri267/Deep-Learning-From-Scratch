A Python implementation of deep learning components from scratch, focusing on fundamental building blocks of neural networks.

## Components

### Neural Network Building Blocks (`dl_src.py`)

1. **Linear Layer**
   - Custom implementation of linear/fully connected layer
   - Supports configurable output nodes
   - Optional bias term
   - Batch processing support
   - Integrated activation functions
   - Weight initialization

2. **Activation Functions**
   - ReLU (Rectified Linear Unit)
     - f(x) = max(0, x)
   - Sigmoid
     - f(x) = 1/(1 + e^(-x))

## Features

- Pure NumPy implementation for better understanding
- Configurable batch size
- Support for different activation functions
- Modular design for easy extension

## Requirements

```
numpy
```

## Usage

### Linear Layer
```python
from dl_src import Linear

# Initialize a linear layer
layer = Linear(
    n_output_nodes=64,  # Number of output features
    bias=True,          # Include bias term
    batch_size=32,      # Batch size for processing
    activation='relu'   # Activation function ('relu' or 'sigmoid')
)

# Forward pass
output = layer.linear_operation(input_data)
```

### Activation Functions
```python
from dl_src import Activations

activations = Activations()

# ReLU activation
relu_output = activations.relu(input_data)

# Sigmoid activation
sigmoid_output = activations.sigmoid(input_data)
```

## Implementation Details

### Linear Layer
- Implements W·X + b operation
- Automatic weight initialization
- Integrated activation function application
- Batch processing support
- Configurable bias term

### Activation Functions
- ReLU: Returns max(0, x) for each input
- Sigmoid: Squashes input to range [0,1]
