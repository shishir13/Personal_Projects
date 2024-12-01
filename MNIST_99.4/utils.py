import torch.nn as nn

def count_parameters(model):
    """
    Counts the total number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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
    param_count = count_parameters(model)
    bn = has_batch_norm(model)
    dropout = has_dropout(model)
    fc = has_fully_connected(model)
    
    print(f"Total Parameters: {param_count}")
    print(f"Batch Normalization Used: {'Yes' if bn else 'No'}")
    print(f"Dropout Used: {'Yes' if dropout else 'No'}")
    print(f"Fully Connected Layers Used: {'Yes' if fc else 'No'}")
