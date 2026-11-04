import os
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))

import torch

model=torch.load(os.path.join(HERE, 'log', 'kelin-v0', 'seed--0', 'QNetwork.125.pt'))
print(model)
