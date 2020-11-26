import sys
import os
import torch
from torch.utils.data import DataLoader
from torch import load
from model import get_model, device
from preprocess import get_sampled_data
from model_parts import MelDataset
from tqdm import tqdm

# Parsing arguments
# First argument must be path to validation data dir, otherwise use default dir
try:
      val_data_path = sys.argv[1]
except IndexError:
      raise BaseException('No validation data path!')

# Second argument - path to pretrained model file,
# otherwise use default
try:
    net_path = sys.argv[2]
except IndexError:
    net_path = os.path.join('pretrained', 'model.ptm')

# Dirs with validation data
val_data_clean_dir = os.path.join(val_data_path, 'clean')
val_data_noisy_dir = os.path.join(val_data_path, 'noisy')

# Get files from validation data
val_data_clean = get_sampled_data(val_data_clean_dir)
val_data_noisy = get_sampled_data(val_data_noisy_dir)

# Make validation dataset
val_dataset = MelDataset(list(zip(val_data_noisy, val_data_clean)), transform=True)
# Make dataloader for model
# We can't use batches, because data has different dimensions
val_loader = DataLoader(dataset=val_dataset, batch_size=1, pin_memory=True)

# Take model
net = get_model()
# Load pretrained model
if os.path.isfile(net_path):
    net.load_state_dict(load(net_path))
else:
    raise BaseException('Neural Network not fitted!')

net.eval()

# Use MSE criterion to evaluate
criterion = torch.nn.MSELoss()

# Evaluate
with torch.no_grad():
    # Total loss variables
    val_loss, noisy_loss = 0., 0.

    loop = tqdm(enumerate(val_loader), total=len(val_loader), leave=True)
    # Validation loop
    for batch_idx, (x, y) in loop:
        # Move data to cuda if possible
        x = x.to(device)

        # Predict and calculate MSE
        y_ = net.predict(x)
        loss = criterion(y_, y)
        val_loss += torch.sum(loss.detach()).item()

        # Calculate MSE of noisy data
        loss = criterion(x.cpu(), y)
        noisy_loss += torch.sum(loss.detach()).item()

        # Update progress bar
        loop.set_description(f'Evaluating')
        loop.set_postfix(noisy_mse=noisy_loss, denoised_mse=val_loss)

print(f'Noisy data MSE: {noisy_loss:.2f}')
print(f'Denoised data MSE: {val_loss:.2f}')
