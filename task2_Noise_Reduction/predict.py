import os
import sys
import numpy as np
from torch import load
from model import get_model
from preprocess import preprocess_to_model

# First argument - file to denoise
try:
    file_path = sys.argv[1]
except IndexError:
    raise BaseException('No file to denoise!')

# Second argument - path to save denoised file,
# otherwise use default
try:
    savefile_path = sys.argv[2]
except IndexError:
    savefile_path = 'predicted.npy'

# Third argument - path to pretrained model file,
# otherwise use default
try:
    net_path = sys.argv[3]
except IndexError:
    net_path = os.path.join('pretrained', 'model.ptm')

# Take model
net = get_model()

# Load pretrained model
if os.path.isfile(net_path):
    net.load_state_dict(load(net_path))
else:
    raise BaseException('Neural Network not fitted!')

net.eval()

# Load data from file and convert to torch.tensor
mel_data = preprocess_to_model(file_path)

# Denoise
denoised_mel_data = net.predict(mel_data).view(-1, 80).numpy()

# Save denoised data to file
np.save(savefile_path, denoised_mel_data)
