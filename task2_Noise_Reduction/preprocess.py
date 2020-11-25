import os
import numpy as np
import scipy
import librosa


# get lists of paths of files
def get_filelist(path) -> list:
    retlist = []
    dirlist = os.listdir(path)
    for directory in dirlist:
        templist = os.listdir(os.path.join(path, directory))
        retlist += list(map(lambda x: os.path.join(path, directory, x), templist))
    return retlist


# Preprocess data to dataset
def load_and_convert(path):
    # load mel-spectrogram
    mel = np.load(path)
    # extract mfcc
    mfcc = librosa.feature.mfcc(y=None, sr=None, S=mel.T, n_mfcc=40)
    # use DCT-2 to extract features from mfcc
    return scipy.fftpack.dct(mfcc, type=2, n=10, norm='ortho').ravel()

def load_mel(path):
    # load mel-spectrogram
    mel = np.load(path)
    return mel

def convert_to_mfcc(mel, n_mfcc=128):
    return librosa.feature.mfcc(S=mel.T, n_mfcc=n_mfcc)


def load_mel_convert_to_mfcc(path, n_mfcc=128):
    # load mel-spectrogram
    mel = np.load(path)
    # convert to mfcc
    mfcc = librosa.feature.mfcc(S=mel.T, n_mfcc=n_mfcc)
    return mfcc