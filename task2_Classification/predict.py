import model
import preprocess
import os.path
from torch import load, from_numpy
import sys

try:
    file_path = sys.argv[1]
except IndexError:
    raise BaseException('No file to predict')

net = model.NeuralNetwork(400, 350, 1)
# Load pretrained model
if os.path.isfile(model.net_path):
    net.load_state_dict(load(model.net_path))
else:
    raise BaseException('Neural Network not fitted!')
net = net.to(model.device)

# Preprocess to net
data_to_predict = from_numpy(preprocess.load_and_convert(file_path))
# Print prediction
print(int(net.predict(data_to_predict).item()))
