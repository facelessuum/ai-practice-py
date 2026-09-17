import torch

DEVICE = torch.device("cude" if torch.cuda.is_available() else "cpu")
