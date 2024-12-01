from utils import count_parameters, has_batch_norm, has_dropout, has_fully_connected
from models.model import MNISTModel

model = MNISTModel()

# Unit test checks
assert count_parameters(model) <= 20000, "Model exceeds 20k parameters!"
assert has_batch_norm(model), "Batch Normalization not used in the model!"
assert has_dropout(model), "Dropout not used in the model!"
assert has_fully_connected(model), "Fully Connected layers not used in the model!"
