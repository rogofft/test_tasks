import model
import preprocess
import os
import sys

# Parsing arguments
# second argument must be path to data dir, otherwise use default dir
try:
      data_path = sys.argv[1]
except IndexError:
      data_path = os.getcwd()

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
train_data_clean = list(map(lambda x: (preprocess.load_and_convert(x), 0), preprocess.get_filelist(train_data_clean_dir)))
train_data_noisy = list(map(lambda x: (preprocess.load_and_convert(x), 1), preprocess.get_filelist(train_data_noisy_dir)))
val_data_clean = list(map(lambda x: (preprocess.load_and_convert(x), 0), preprocess.get_filelist(val_data_clean_dir)))
val_data_noisy = list(map(lambda x: (preprocess.load_and_convert(x), 1), preprocess.get_filelist(val_data_noisy_dir)))

print(f'Train data: clean - {len(train_data_clean)}, noisy - {len(train_data_noisy)}\n'
      f'Val data: clean - {len(val_data_clean)} , noisy - {len(val_data_noisy)}')

# Make train and val datasets
train_dataset = model.MelDataset(train_data_clean + train_data_noisy, transform=True)
val_dataset = model.MelDataset(val_data_clean + val_data_noisy, transform=True)

# Make NN model
print('Using', str(model.device))
net = model.NeuralNetwork(400, 350, 1).to(model.device)

# Fit the model and save best score parameters
print('Start fitting...')
net.fit(train_dataset, 20, lr=0.001, batch_size=10, val_dataset=val_dataset)
