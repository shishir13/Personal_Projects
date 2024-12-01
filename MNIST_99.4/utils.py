import torch.nn as nn

def count_parameters(model):
    """
    Counts the total number of trainable parameters in the model.
    """
    total_params = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            param_count = parameter.numel()
            print(f"{name}: {param_count:,} parameters")
            total_params += param_count
    print(f"\nTotal trainable parameters: {total_params:,}")
    return total_params

def has_batch_norm(model):
    """
    Checks if the model contains Batch Normalization layers.
    """
    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm2d):
            return True
    return False

def has_dropout(model):
    """
    Checks if the model contains Dropout layers.
    """
    for layer in model.modules():
        if isinstance(layer, nn.Dropout):
            return True
    return False

def has_fully_connected(model):
    """
    Checks if the model contains Fully Connected (Linear) layers.
    """
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            return True
    return False

def print_model_summary(model):
    """
    Prints a summary of the model including parameter count
    and layer types used.
    """
    print("Model Summary:")
    print("-" * 40)
    param_count = count_parameters(model)
    bn = has_batch_norm(model)
    dropout = has_dropout(model)
    fc = has_fully_connected(model)
    
    print(f"\nArchitecture Requirements:")
    print(f"- Total Parameters: {param_count:,} {'[PASS]' if param_count <= 20000 else '[FAIL]'}")
    print(f"- Batch Normalization: {'Yes [PASS]' if bn else 'No [FAIL]'}")
    print(f"- Dropout: {'Yes [PASS]' if dropout else 'No [FAIL]'}")
    print(f"- Fully Connected Layers: {'Yes [PASS]' if fc else 'No [FAIL]'}")
    print("-" * 40)
