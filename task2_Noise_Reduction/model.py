import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score

torch.manual_seed(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#net_path = os.path.join('pretrained', 'net_config.ptn')


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv1d(1, 18, 7)
        self.bn1 = nn.BatchNorm1d(18)
        self.conv2 = nn.Conv1d(18, 30, 4)
        self.bn2 = nn.BatchNorm1d(30)
        self.conv3 = nn.Conv1d(30, 8, 7)
        self.bn3 = nn.BatchNorm1d(8)

        self.conv4 = nn.Conv1d(8, 18, 7)
        self.bn4 = nn.BatchNorm1d(18)
        self.conv5 = nn.Conv1d(18, 30, 4)
        self.bn5 = nn.BatchNorm1d(30)
        self.conv6 = nn.Conv1d(30, 8, 7)
        self.bn6 = nn.BatchNorm1d(8)

        self.conv7 = nn.Conv1d(8, 18, 7)
        self.bn7 = nn.BatchNorm1d(18)
        self.conv8 = nn.Conv1d(18, 30, 4)
        self.bn8 = nn.BatchNorm1d(30)
        self.conv9 = nn.Conv1d(30, 8, 7)
        self.bn9 = nn.BatchNorm1d(8)

        self.conv10 = nn.Conv1d(8, 18, 7)
        self.bn10 = nn.BatchNorm1d(18)
        self.conv11 = nn.Conv1d(18, 30, 4)
        self.bn11 = nn.BatchNorm1d(30)
        self.conv12 = nn.Conv1d(30, 8, 7)
        self.bn12 = nn.BatchNorm1d(8)

        self.conv13 = nn.Conv1d(8, 18, 7)
        self.bn13 = nn.BatchNorm1d(18)
        self.conv14 = nn.Conv1d(18, 30, 4)
        self.bn14 = nn.BatchNorm1d(30)
        self.conv15 = nn.Conv1d(30, 8, 7)
        self.bn15 = nn.BatchNorm1d(8)
        self.conv16 = nn.Conv1d(8, 18, 5)

        self.conv17 = nn.Conv1d(18, 80, 1)
        self.drop = nn.Dropout(p=0.3)

    def forward(self, x):
        x = torch.relu(self.drop(self.bn3(self.conv3(torch.relu(self.drop(self.bn2(self.conv2(torch.relu(self.drop(self.bn1(self.conv1(x))))))))))))
        x = torch.relu(self.drop(self.bn6(self.conv6(torch.relu(self.drop(self.bn5(self.conv5(torch.relu(self.drop(self.bn4(self.conv4(x))))))))))))
        x = torch.relu(self.drop(self.bn9(self.conv9(torch.relu(self.drop(self.bn8(self.conv8(torch.relu(self.drop(self.bn7(self.conv7(x))))))))))))
        x = torch.relu(self.drop(self.bn12(self.conv12(torch.relu(self.drop(self.bn11(self.conv11(torch.relu(self.drop(self.bn10(self.conv10(x))))))))))))
        x = torch.relu(self.drop(self.bn15(self.conv15(torch.relu(self.drop(self.bn14(self.conv14(torch.relu(self.drop(self.bn13(self.conv13(x))))))))))))
        x = self.conv17(torch.relu(self.conv16(x)))
        return x.view(-1, 1, 80)

    def predict(self, x):
        x = x.to(device)
        y_ = torch.cat(list(map(lambda r: self.forward(x[:1, r:r+1, :]), range(x.size()[1]))), 1)
        return y_.cpu().detach()

    def fit(self, dataset, epochs, lr=0.0001, batch_size=1, val_dataset=None, model_save_path='pretrained/model.ptm'):

        # Check for directory to save is exists
        #if not os.path.exists(model_save_path):
        #    os.makedirs(model_save_path)

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        # Loaders for evaluation
        trainloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        if val_dataset:
            evalloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
        else:
            evalloader = None

        # Var for summary evaluating loss
        eval_loss = 0.

        # Early stop detector
        early_stopping = EarlyStopping(num_to_stop=7)

        # Train network
        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.
            loop = tqdm(enumerate(trainloader), total=len(trainloader), leave=False)
            for batch_idx, (x, y) in loop:
                # Move data to cuda if possible
                x, y = x.to(device), y.to(device)

                # Forward
                y_ = self.forward(x)
                loss = criterion(y_, y)
                epoch_loss += torch.sum(loss.detach()).item()

                # Backward
                optimizer.zero_grad()
                loss.backward()

                # Gradient step
                optimizer.step()

                # Update progress bar
                loop.set_description(f'Epoch [{epoch+1}/{epochs}]')
                loop.set_postfix(loss=epoch_loss, mse=eval_loss)

            # Evaluate
            with torch.no_grad():
                self.eval()
                eval_loss = 0.
                loop = tqdm(enumerate(evalloader), total=len(evalloader), leave=True)

                for batch_idx, (x, y) in loop:
                    # Move data to cuda if possible
                    x, y = x.to(device), y.to(device)

                    # Forward
                    y_ = self.forward(x)
                    loss = criterion(y_, y)
                    eval_loss += torch.sum(loss.detach()).item()
                    loop.set_description(f'Epoch [{epoch + 1}/{epochs}]')
                    loop.set_postfix(loss=epoch_loss, mse=eval_loss)

            # Check for early stopping
            if early_stopping(eval_loss):
                # Exit from train loop
                break

            # Check for best score
            if early_stopping.is_best_score():
                # Save the model
                torch.save(self.state_dict(), model_save_path)
                print('Saving')

        self.eval()


class MelDataset(Dataset):
    def __init__(self, data, transform=False):
        self.data = data
        self.length = len(data)
        self.transform = transform

    def __getitem__(self, idx):
        data, target = self.data[idx][0], self.data[idx][1]
        if self.transform:
            data, target = torch.from_numpy(data), torch.from_numpy(target)
        return data, target

    def __len__(self):
        return self.length


class EarlyStopping:
    """ Class for early stopping the model during fitting.
        Use this for scores, where lower result is better than higher.
        num_to_stop - number of iterations with not improving prefomance.
        returns True if you should stop training. """

    def __init__(self, num_to_stop=7):
        self.best_score = None
        self.lower_score_counter = 0
        self.num_to_stop = num_to_stop

    def __call__(self, loss) -> bool:
        # First call
        if not self.best_score:
            self.best_score = loss
            return False
        # Check for best score
        else:
            if loss < self.best_score:
                # New best score achieved
                self.best_score = loss
                self.lower_score_counter = 0
            else:
                # Bad score
                self.lower_score_counter += 1

            # Check for stopping
            if self.lower_score_counter >= self.num_to_stop:
                return True
            else:
                return False

    def is_best_score(self):
        """ Use this function to decide to save the model
            If a best score was achieved, counter will turn to 0 """
        return self.lower_score_counter == 0

'''
# Evaluating
            with torch.no_grad():
                self.eval()

                x, y = next(iter(eval_trainloader))
                train_acc = accuracy_score(y, self.predict(x))

                if val_dataset:
                    x, y = next(iter(eval_valloader))
                    val_acc = accuracy_score(y, self.predict(x))

                # Save best model

                if val_dataset:
                    score = val_acc
                else:
                    score = train_acc
                if self.best_acc_score < score:
                    self.best_acc_score = score
                    # Save new weights
                    torch.save(self.state_dict(), net_path)

                self.train()
                
            if val_dataset:
                print(
                    f'Epoch: {epoch + 1}/{epochs}, loss: {epoch_loss:5.4f}, '
                    f'train acc: {train_acc:2.4f}, val_acc: {val_acc:2.4f}')
            else:
                print(f'Epoch: {epoch + 1}/{epochs}, loss: {epoch_loss:5.4f}, train acc: {train_acc:2.4f}')'''