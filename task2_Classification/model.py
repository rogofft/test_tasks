import torch
import torch.nn as nn
import torch.optim as optim
import os

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score

torch.manual_seed(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net_path = os.path.join('pretrained', 'net_config.ptn')


class NeuralNetwork(nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super(NeuralNetwork, self).__init__()
        self.linear1 = nn.Linear(n_in, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_out)
        self.drop = nn.Dropout(p=0.5)
        self.best_acc_score = 0.

    def forward(self, x):
        x = torch.relu(self.drop(self.linear1(x)))
        x = torch.sigmoid(self.linear2(x))
        return x

    def predict(self, x):
        x = x.to(device)
        x = self.forward(x)
        return (x.cpu().detach() >= 0.5) * 1.

    def fit(self, dataset, epochs, lr=0.01, batch_size=1, val_dataset=None):
        self.train()
        trainloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        optimizer = optim.RMSprop(self.parameters(), lr=lr)
        criterion = nn.BCELoss()

        # Loaders for evaluation
        eval_trainloader = DataLoader(dataset=dataset, batch_size=len(dataset), shuffle=False, pin_memory=True)
        if val_dataset:
            eval_valloader = DataLoader(dataset=val_dataset, batch_size=len(val_dataset), shuffle=False,
                                        pin_memory=True)
        else:
            eval_valloader = None

        for epoch in range(epochs):
            epoch_loss = 0
            for x, y in trainloader:
                x, y = x.to(device), y.to(device)
                y_ = self.forward(x)
                loss = criterion(y_, y)
                epoch_loss += loss.detach().item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

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
                print(f'Epoch: {epoch + 1}/{epochs}, loss: {epoch_loss:5.4f}, train acc: {train_acc:2.4f}')

        self.eval()


class MelDataset(Dataset):
    def __init__(self, data, transform=False):
        self.data = data
        self.length = len(data)
        self.transform = transform

    def __getitem__(self, idx):
        data, target = self.data[idx][0], self.data[idx][1]
        if self.transform:
            data, target = torch.from_numpy(data), torch.FloatTensor([target])
        return data, target

    def __len__(self):
        return self.length
