import torch
import torch.nn as nn
import torch.optim as optim
from model_parts import EarlyStopping
from torch.utils.data import DataLoader
from tqdm import tqdm

# Use manual seed to reproduce results
#torch.manual_seed(1)
# Use cuda if it's allowed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_model():
    """ Use this function to get the model. """
    return NeuralNetwork().to(device)


class NeuralNetwork(nn.Module):
    """ Model for noise reduction.
        Used convolutional net with 2 dense layers in the end.
        Net saves the best (based on validation) parameters and has early stopping while training.
        To use as noise reduction filter use 'predict' function. """

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        self.lrelu1 = nn.LeakyReLU(negative_slope=0.2)

        self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        self.lrelu2 = nn.LeakyReLU(negative_slope=0.2)

        self.conv3 = nn.Conv1d(64, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(2)
        self.lrelu3 = nn.LeakyReLU(negative_slope=0.2)

        self.conv4 = nn.Conv1d(64, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm1d(128)
        self.pool4 = nn.MaxPool1d(2)
        self.lrelu4 = nn.LeakyReLU(negative_slope=0.2)

        self.conv5 = nn.Conv1d(128, 128, 3, padding=1)
        self.bn5 = nn.BatchNorm1d(128)
        self.pool5 = nn.MaxPool1d(2)
        self.lrelu5 = nn.LeakyReLU(negative_slope=0.2)

        self.conv6 = nn.Conv1d(128, 256, 3, padding=1)
        self.bn6 = nn.BatchNorm1d(256)
        self.pool6 = nn.AvgPool1d(2)
        self.lrelu6 = nn.LeakyReLU(negative_slope=0.2)

        self.flatten = nn.Flatten()
        self.linear7 = nn.Linear(256, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.lrelu7 = nn.LeakyReLU()
        self.linear8 = nn.Linear(256, 80)

    def forward(self, x):
        x = x.view(-1, 1, 80)
        x = self.pool1(self.lrelu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.lrelu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.lrelu3(self.bn3(self.conv3(x))))
        x = self.pool4(self.lrelu4(self.bn4(self.conv4(x))))
        x = self.pool5(self.lrelu5(self.bn5(self.conv5(x))))
        x = self.pool6(self.lrelu6(self.bn6(self.conv6(x))))

        x = self.flatten(x)

        x = self.lrelu7(self.bn7(self.linear7(x)))
        x = self.linear8(x)
        return x.view(1, -1, 80)

    def predict(self, x):
        """ Use this function to reduce noise on data. """
        x = x.to(device)
        y_ = self.forward(x)
        return y_.detach().cpu()

    def fit(self, dataset, val_dataset, epochs, lr=0.0001, batch_size=1024, model_save_path='pretrained/model.ptm'):
        """ Function to fit model.
            :dataset - train dataset (use class MelDataset)
            :val_dataset - validation dataset
            :model_save_path - path to file to save model on checkpoints """

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        # Loaders for train and evaluation
        trainloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        evalloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

        # Early stop detector
        early_stopping = EarlyStopping(num_to_stop=5)

        # Train network
        for epoch in range(epochs):
            self.train()
            eval_loss = 0.
            train_loss = 0.
            loop = tqdm(enumerate(trainloader), total=len(trainloader), leave=False)

            # Train loop
            for batch_idx, (x, y) in loop:
                # Move data to cuda if possible
                x, y = x.to(device), y.to(device)

                # Forward
                y_ = self.forward(x)
                loss = criterion(y_, y)
                train_loss += torch.sum(loss.detach()).item()

                # Backward
                optimizer.zero_grad()
                loss.backward()

                # Gradient step
                optimizer.step()

                # Update progress bar
                loop.set_description(f'Epoch [{epoch+1}/{epochs}] Training')
                loop.set_postfix(train_loss=train_loss, val_mse=eval_loss)

            # Evaluate
            with torch.no_grad():
                self.eval()
                loop = tqdm(enumerate(evalloader), total=len(evalloader), leave=True)

                # Evaluating loop
                for batch_idx, (x, y) in loop:
                    # Move data to cuda if possible
                    x, y = x.to(device), y.to(device)

                    # Forward
                    y_ = self.forward(x)
                    loss = criterion(y_, y)
                    eval_loss += torch.sum(loss.detach()).item()

                    # Update progress bar
                    loop.set_description(f'Epoch [{epoch + 1}/{epochs}] Evaluating')
                    loop.set_postfix(train_loss=train_loss, val_mse=eval_loss)

            # Check for early stopping
            if early_stopping(eval_loss):
                # Exit from train loop
                break

            # Check for best score
            if early_stopping.is_best_score():
                # Save the model
                torch.save(self.state_dict(), model_save_path)

        self.eval()
