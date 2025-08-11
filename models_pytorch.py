import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model architectures
class MLPRegressor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.Sigmoid(),
            nn.Linear(256, 128),
            nn.Sigmoid(),
            nn.Linear(128, 64),
            nn.Sigmoid(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        out = self.net(x)
        return out.view(-1)  # flatten to shape (batch,)

class MLPClassifier(nn.Module):
    def __init__(self, input_size, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.net(x)

class CNNRegressor(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        c, l = input_shape
        self.conv = nn.Sequential(
            nn.Conv1d(c, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        conv_out_size = (l // 8) * 64
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(conv_out_size, 1)
        )
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x).squeeze(-1)

class CNNClassifier(nn.Module):
    def __init__(self, input_shape, num_classes=4):
        super().__init__()
        c, l = input_shape
        self.conv = nn.Sequential(
            nn.Conv1d(c, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        conv_out_size = (l // 8) * 64
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(conv_out_size, num_classes)
        )
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class LSTMRegressor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, 100, batch_first=True)
        self.dropout1 = nn.Dropout(0.3)
        self.lstm2 = nn.LSTM(100, 50, batch_first=True)
        self.dropout2 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(50, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        out = self.relu(self.fc1(out[:, -1, :]))
        out = self.fc2(out)
        return out.squeeze(-1)

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, num_classes=4):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, 100, batch_first=True)
        self.dropout1 = nn.Dropout(0.3)
        self.lstm2 = nn.LSTM(100, 50, batch_first=True)
        self.dropout2 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(50, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        out = self.relu(self.fc1(out[:, -1, :]))
        out = self.fc2(out)
        return out

    
def random_forest_reg():
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor()
    return rf



def linear_regression():
    
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
 
    return lr



def svc():
    from sklearn.svm import SVC
    svm =SVC(C=100,gamma=0.001,kernel='rbf')

    return svm

def random_forest_clf():
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=500,criterion='entropy')

    return rf
