import pandas as pd
import numpy as np
import os
import re
import collections, time
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn import feature_extraction
import torch, torchtext
import torch.nn as nn
import torch.functional as F
from torch.utils.data import TensorDataset, DataLoader

# Define hyperparameters
EPOCH_MAX = 2000
EPOCH_LOG = 100
OPTIMIZER_PARAMS = {'lr': 1}
DATALOADER_PARAMS = {'batch_size': 1000, 'shuffle': True}
USE_CUDA = torch.cuda.is_available()
RANDOM_SEED = 77

class MyGRU(nn.Module):
    def __init__(self, embedding, gru_hidden_size=100, gru_num_layer=1, output_size=18):
        super().__init__()
        self.embed = nn.Embedding.from_pretrained(embedding, freeze=True)
        self.drop = nn.Dropout(0.2)
        self.rnn = nn.GRU(input_size=embedding.shape[-1], hidden_size=gru_hidden_size, num_layers=gru_num_layer, batch_first=True)
        self.classifier = nn.Linear(gru_num_layer, output_size)

        self.embed.weight.requires_grad = False
        self.classifier.weight.data.uniform_(-0.5, 0.5)
        self.classifier.bias.data.zero_()

    def forward(self, indices):
        vectors = self.embed(indices)
        vectors = self.drop(vectors)
        outputs, hidden = self.rnn(vectors)
        classifier_outputs = self.classifier(outputs[:, -1]) # Use output of the last sentence

        return classifier_outputs

from preprocess import preprocess
data = preprocess()
data.make_dataframe()

X_train, X_test= train_test_split(data.review, test_size=0.2, random_state=RANDOM_SEED, shuffle=True)
y_train, y_test= train_test_split(data.target, test_size=0.2, random_state=RANDOM_SEED, shuffle=True)
X_train = np.array(X_train)
X_test = np.array(X_test)

from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
yt = mlb.fit_transform(y_train)
