import torch
from torch.utils.data import Dataset


class MelDataset(Dataset):
    """ Dataset to store [1, 80] samples of mel-spectrogram """
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

    def __init__(self, num_to_stop=5):
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
