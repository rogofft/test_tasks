import model
import preprocess
import os
import sys
import torch
import librosa
from model import device
import soundfile
import numpy as np
import time

from torch.utils.data import DataLoader

# Parsing arguments
# second argument must be path to data dir, otherwise use default dir
#try:
#      data_path = sys.argv[1]
#except IndexError:
#      data_path = os.getcwd()

data_path = 'E:\MachineLearningCourse\Goznak_ML_Tasks\data'
net_path = os.path.join('pretrained', 'model.ptm')
# train directory path
train_data_dir = 'train'
# validation directory path
val_data_dir = 'val'

# Dirs
train_data_clean_dir = os.path.join(data_path, train_data_dir, 'clean')
train_data_noisy_dir = os.path.join(data_path, train_data_dir, 'noisy')
val_data_clean_dir = os.path.join(data_path, val_data_dir, 'clean')
val_data_noisy_dir = os.path.join(data_path, val_data_dir, 'noisy')
# print(train_data_clean_dir, train_data_noisy_dir, val_data_clean_dir, val_data_noisy_dir, sep='\n')

# Preprocessing data
print('Start preprocessing data (may take a few minutes)...')
train_data_clean = np.concatenate([np.load(path).astype(np.single).reshape(-1, 1, 80) for path in preprocess.get_filelist(train_data_clean_dir)[2900:3000]])
train_data_noisy = np.concatenate([np.load(path).astype(np.single).reshape(-1, 1, 80) for path in preprocess.get_filelist(train_data_noisy_dir)[2900:3000]])
val_data_clean   = np.concatenate([np.load(path).astype(np.single).reshape(-1, 1, 80) for path in preprocess.get_filelist(val_data_clean_dir)[:200]])
val_data_noisy   = np.concatenate([np.load(path).astype(np.single).reshape(-1, 1, 80) for path in preprocess.get_filelist(val_data_noisy_dir)[:200]])

print(train_data_clean.shape)

print(f'Train data: clean - {len(train_data_clean)}, noisy - {len(train_data_noisy)}\n'
      f'Val data: clean - {len(val_data_clean)} , noisy - {len(val_data_noisy)}')

# Make train and val datasets
train_dataset = model.MelDataset(list(zip(train_data_noisy, train_data_clean)), transform=True)
val_dataset = model.MelDataset(list(zip(val_data_noisy, val_data_clean)), transform=True)


net = model.NeuralNetwork().to(device)
if os.path.exists(net_path):
      net.load_state_dict(torch.load(net_path))
net.fit(train_dataset, epochs=40, lr=0.0001, batch_size=1024, val_dataset=val_dataset, model_save_path=net_path)
#torch.save(net.state_dict(), net_path)
net.eval()


'''test_loader = DataLoader(val_dataset, batch_size=1, pin_memory=True)

for x, y in test_loader:

      print(x.size())
      f, axs = plt.subplots(3, 1)

      start = time.time()
      y_ = net.predict(x.to(device))
      print(time.time() - start)

      noisy = x.numpy()
      denoisy = y_.numpy()
      clean = y.numpy()


      axs[0].set_title('Noisy')
      axs[0].imshow(noisy.T)
      axs[1].set_title('Model Predict')
      axs[1].imshow(denoisy.T)
      axs[2].set_title('Clean')
      axs[2].imshow(clean.T)
      plt.tight_layout()
      plt.show()'''


