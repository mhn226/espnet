import torch

entrypoints = torch.hub.list('pytorch/vision', force_reload=True)
print(entrypoints)