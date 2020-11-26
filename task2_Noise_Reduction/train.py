import os
import sys

from preprocess import get_sampled_data
from model import get_model
from model_parts import MelDataset

# Parsing arguments
# First argument must be path to data dir, otherwise use default dir
try:
      data_path = sys.argv[1]
except IndexError:
      data_path = os.getcwd()

# Path to save trained model
net_path = os.path.join('pretrained', 'model.ptm')

# Define dirs
train_data_clean_dir = os.path.join(data_path, 'train', 'clean')
train_data_noisy_dir = os.path.join(data_path, 'train', 'noisy')
val_data_clean_dir = os.path.join(data_path, 'val', 'clean')
val_data_noisy_dir = os.path.join(data_path, 'val', 'noisy')

# Load the data samples
train_data_clean = get_sampled_data(train_data_clean_dir)
train_data_noisy = get_sampled_data(train_data_noisy_dir)
val_data_clean = get_sampled_data(val_data_clean_dir)
val_data_noisy = get_sampled_data(val_data_noisy_dir)

print(f'Total Train data samples: clean - {len(train_data_clean)}, noisy - {len(train_data_noisy)}\n'
      f'Total Val data samples: clean - {len(val_data_clean)} , noisy - {len(val_data_noisy)}')

# Make train and val datasets
train_dataset = MelDataset(list(zip(train_data_noisy, train_data_clean)), transform=True)
val_dataset = MelDataset(list(zip(val_data_noisy, val_data_clean)), transform=True)

# Take a model
net = get_model()

# Training
net.fit(train_dataset, val_dataset=val_dataset, epochs=40, lr=0.0001, batch_size=1, model_save_path=net_path)

print('Model fitted successfully!')
