import torch
from torch import nn
import torch.nn.functional as F


class HARModelWithLSTM(nn.Module):

    def __init__(self, n_hidden=128, n_layers=1, n_filters=64,
                 n_classes=32, filter_size=(5, 1), drop_prob=0.5):
        super(HARModelWithLSTM, self).__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_filters = n_filters
        self.n_classes = n_classes
        self.filter_size = filter_size

        self.conv1 = nn.Conv2d(1, n_filters, filter_size)
        self.conv2 = nn.Conv2d(n_filters, n_filters, filter_size)
        self.conv3 = nn.Conv2d(n_filters, n_filters, filter_size)
        self.conv4 = nn.Conv2d(n_filters, n_filters, filter_size)

        self.lstm1 = nn.LSTM(n_filters * 3, n_hidden, n_layers, batch_first=True)
        self.lstm2 = nn.LSTM(n_hidden, n_hidden, n_layers, batch_first=True)

        self.fc = nn.Linear(n_hidden * 8, n_classes)

        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        # (b, 1, 24, 3) -> (b, 64, 8 , 3)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # (b, 64, 8 , 3) -> (b , 8 , 64, 3)
        x = x.permute(0, 2, 1, 3)
        x = x.contiguous()
        # (b, 8, 64, 3) -> (b, 8, 192)
        x = x.view(-1, 8, 64 * 3)

        # (b, 8, 192) -> (b, 8, 128)
        x = self.dropout(x)
        x, (h_n, c_n) = self.lstm1(x)

        # (b, 8, 128) -> (b, 8, 128)
        x = self.dropout(x)
        x, (h_n, c_n) = self.lstm2(x)

        # (b, 8, 128) -> (b, 1024)
        x = x.contiguous().view(-1, 8 * 128)
        x = self.dropout(x)

        # (b, 1024) -> (b, 32)
        # out = F.softmax(self.fc(x), dim=1)

        return x
