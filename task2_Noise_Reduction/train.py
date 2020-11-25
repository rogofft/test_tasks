import model
import preprocess
import os
import sys
import torch
import librosa
from model import device
import soundfile
import numpy as np

# Parsing arguments
# second argument must be path to data dir, otherwise use default dir
#try:
#      data_path = sys.argv[1]
#except IndexError:
#      data_path = os.getcwd()

data_path = 'E:\MachineLearningCourse\Goznak_ML_Tasks\data'
net_path = os.path.join('pretrained', 'net_config.ptn')
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
train_data_clean = list(map(lambda x: preprocess.load_mel(x), preprocess.get_filelist(train_data_clean_dir)[:10]))
train_data_noisy = list(map(lambda x: preprocess.load_mel(x), preprocess.get_filelist(train_data_noisy_dir)[:10]))
val_data_clean = list(map(lambda x: preprocess.load_mel(x), preprocess.get_filelist(val_data_clean_dir)[:50]))
val_data_noisy = list(map(lambda x: preprocess.load_mel(x), preprocess.get_filelist(val_data_noisy_dir)[:50]))

print(f'Train data: clean - {len(train_data_clean)}, noisy - {len(train_data_noisy)}\n'
      f'Val data: clean - {len(val_data_clean)} , noisy - {len(val_data_noisy)}')

# Make train and val datasets
train_dataset = model.MelDataset(list(zip(train_data_noisy, train_data_clean)), transform=True)
val_dataset = model.MelDataset(list(zip(val_data_noisy, val_data_clean)), transform=True)

# Make NN model
#print('Using', str(model.device))
#net = model.NeuralNetwork(400, 350, 1).to(model.device)

# Fit the model and save best score parameters
#print('Start fitting...')
#net.fit(train_dataset, 20, lr=0.001, batch_size=10, val_dataset=val_dataset)

#import librosa
import matplotlib.pyplot as plt
import seaborn as sns





net = model.NeuralNetwork(80, 1024, 1024, 80).to(device)
net.load_state_dict(torch.load(net_path))
#net.fit(train_dataset, epochs=2, lr=0.0001)
#torch.save(net.state_dict(), net_path)
net.eval()

mse_denoisy = []
mse_noisy = []

for i in range(len(val_dataset)):
      print(i)
      x, y = val_dataset[i]
      f, axs = plt.subplots(3, 1)

      y_ = net.predict(x.to(device))

      noisy = x.numpy()
      denoisy = y_.numpy()
      clean = y.numpy()

      mse_denoisy.append(np.power(clean - denoisy, 2).mean())
      mse_noisy.append(np.power(clean - noisy, 2).mean())

      axs[0].set_title('Noisy')
      axs[0].imshow(noisy)
      axs[1].set_title('Model Predict')
      axs[1].imshow(denoisy)
      axs[2].set_title('Clean')
      axs[2].imshow(clean)
      plt.tight_layout()
      plt.show()
print(np.mean(mse_denoisy))
print(np.mean(mse_noisy))


