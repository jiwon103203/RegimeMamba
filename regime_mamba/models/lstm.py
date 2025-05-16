'''
Stacked LSTM Model Class Implementation
Uses a multi-layer structure with stacked LSTM layers
Consists of 4 LSTM layers (50 units each) with 0.1 dropout after each layer,
and a single output neuron, using Adam optimizer and MSE loss function
Input sequence length is 60, batch size is 2048
Args:
    input_dim (int): Input feature dimension
    hidden_dim (int): LSTM hidden state dimension
    num_layers (int): Number of LSTM layers
    output_dim (int): Output feature dimension
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd

class StackedLSTM(nn.Module):
    '''
    Stacked LSTM Model Class
    Uses a multi-layer structure with stacked LSTM layers
    Consists of 4 LSTM layers (50 units each) with 0.1 dropout after each layer,
    and a single output neuron, using Adam optimizer and MSE loss function
    Input sequence length is 60, batch size is 2048
    Args:
        input_dim (int): Input feature dimension
        hidden_dim (int): LSTM hidden state dimension
        num_layers (int): Number of LSTM layers
        output_dim (int): Output feature dimension
    '''
    def __init__(self, input_dim, hidden_dim=32, num_layers=8, output_dim=1):
        super(StackedLSTM, self).__init__()
        
        # Initialize LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Initialize dropout layer
        self.dropout = nn.Dropout(0.1)
        
        # Initialize output layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, return_hidden=False):
        # Pass through LSTM layer
        lstm_out, _ = self.lstm(x)
        
        # Select last sequence output
        lstm_out = lstm_out[:, -1, :]
        
        # Apply dropout
        out = self.dropout(lstm_out)
        
        # Pass through output layer
        out = self.fc(out)

        if return_hidden:
            return out, lstm_out
        return out
    
    def train_for_window(self, train_start, train_end, data, valid_window, outputdir):
        '''
        Set up data window for model training
        Args:
            train_start (int): Training data start index
            train_end (int): Training data end index
            data (torch.Tensor): Input data
            valid_window (int): Validation window size
            outputdir (str): Model save path
        '''
        self.train_start = train_start
        self.train_end = train_end
        self.valid_window = valid_window
        
        # Split training and validation data
        self.train_data = data[(data['Date'] >= train_start) & (data['Date'] <= train_end)].copy()
        self.train_data = self.train_data[['dd_10','sortino_20','sortino_60','dollar_index', 'target_returns_1']]

        self.train_sequence_x = []
        self.train_sequence_y = []
        for i in range(1, len(self.train_data)):
            if i < 60:
                continue
            self.train_sequence_x.append(self.train_data.iloc[i-60:i][['dd_10','sortino_20','sortino_60','dollar_index']].values)
            self.train_sequence_y.append(self.train_data.iloc[i]['target_returns_1'])
        self.train_sequence_x = torch.tensor(self.train_sequence_x, dtype=torch.float32)
        self.train_sequence_y = torch.tensor(self.train_sequence_y, dtype=torch.float32).unsqueeze(1)  # Convert to 1D

        self.valid_end = str(datetime.strptime(train_end, '%Y-%m-%d') + relativedelta(years=valid_window))
        self.valid_data = data[(data['Date'] >= train_end) & (data['Date'] <= self.valid_end)].copy()
        self.valid_data = self.valid_data[['dd_10','sortino_20','sortino_60','dollar_index', 'target_returns_1']]
        self.valid_sequence_x = []
        self.valid_sequence_y = []
        for i in range(1, len(self.valid_data)):
            if i < 60:
                continue
            self.valid_sequence_x.append(self.valid_data.iloc[i-60:i][['dd_10','sortino_20','sortino_60','dollar_index']].values)
            self.valid_sequence_y.append(self.valid_data.iloc[i]['target_returns_1'])
        self.valid_sequence_x = torch.tensor(self.valid_sequence_x, dtype=torch.float32)
        self.valid_sequence_y = torch.tensor(self.valid_sequence_y, dtype=torch.float32).unsqueeze(1)

        # Create TensorDataset
        self.train_dataset = TensorDataset(self.train_sequence_x, self.train_sequence_y)
        self.valid_dataset = TensorDataset(self.valid_sequence_x, self.valid_sequence_y)

        train_dataloader = DataLoader(self.train_dataset, batch_size=2048, shuffle=False)
        valid_dataloader = DataLoader(self.valid_dataset, batch_size=2048, shuffle=False)

        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        patience = 30
        best_loss = float('inf')
        best_model = None
        self.to('cuda')
        #self.dropout.to('cuda')
        #self.fc.to('cuda')

        for epoch in range(1000):
            for i, (x, y) in enumerate(train_dataloader):
                # Train model
                x = x.to('cuda')
                y = y.to('cuda')
                self.train()
                optimizer.zero_grad()
                output = self.forward(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                
                if i % 100 == 0:
                    print(f'Epoch [{epoch+1}/1000], Loss: {loss.item():.5f}')

            # Evaluate model on validation data
            self.eval()
            valid_loss = 0
            with torch.no_grad():
                for x, y in valid_dataloader:
                    x = x.to('cuda')
                    y = y.to('cuda')
                    output = self.forward(x)
                    loss = criterion(output, y)
                    valid_loss += loss.item()
                valid_loss /= len(valid_dataloader)
                print(f'Validation Loss: {valid_loss:.5f}')
                # Check early stopping condition
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    best_model = self.state_dict()
                    patience = 30
                else:
                    patience -= 1
                    if patience == 0:
                        print('Early stopping')
                        break
        # Save best model
        torch.save(best_model, f'./{outputdir}/best_model.pth')
        # Load model
        print('Best model loaded')