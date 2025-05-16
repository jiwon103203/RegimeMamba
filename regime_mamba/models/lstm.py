'''
Stacked LSTM 모델 Class 구현
여러 LSTM 레이어를 쌓은 다층 구조가 사용
4개의 LSTM 레이어(각각 50개 유닛)와 각 레이어 뒤에 0.1 드롭아웃 레이어를 배치하며,
출력층에는 단일 뉴런을 사용하고 Adam 최적화 알고리즘과 MSE 손실 함수를 적용
입력 시퀀스의 길이는 60, 배치 크기는 2048
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
    Stacked LSTM 모델 Class
    여러 LSTM 레이어를 쌓은 다층 구조가 사용
    4개의 LSTM 레이어(각각 50개 유닛)와 각 레이어 뒤에 0.1 드롭아웃 레이어를 배치하며,
    출력층에는 단일 뉴런을 사용하고 Adam 최적화 알고리즘과 MSE 손실 함수를 적용
    입력 시퀀스의 길이는 60, 배치 크기는 2048
    Args:
        input_dim (int): 입력 특성의 차원
        hidden_dim (int): LSTM의 은닉 상태 차원
        num_layers (int): LSTM 레이어의 수
        output_dim (int): 출력 특성의 차원
    '''
    def __init__(self, input_dim, hidden_dim=32, num_layers=8, output_dim=1):
        super(StackedLSTM, self).__init__()
        
        # LSTM 레이어 초기화
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # 드롭아웃 레이어 초기화
        self.dropout = nn.Dropout(0.1)
        
        # 출력층 초기화
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, return_hidden=False):
        # LSTM 레이어 통과
        lstm_out, _ = self.lstm(x)
        
        # 마지막 시퀀스 출력 선택
        lstm_out = lstm_out[:, -1, :]
        
        # 드롭아웃 적용
        out = self.dropout(lstm_out)
        
        # 출력층 통과
        out = self.fc(out)

        if return_hidden:
            return out, lstm_out
        return out
    
    def train_for_window(self, train_start, train_end, data, valid_window, outputdir):
        '''
        모델 훈련을 위한 데이터 윈도우 설정
        Args:
            train_start (int): 훈련 데이터 시작 인덱스
            train_end (int): 훈련 데이터 종료 인덱스
            data (torch.Tensor): 입력 데이터
            valid_window (int): 검증 윈도우 크기
            outputdir (str): 모델 저장 경로
        '''
        self.train_start = train_start
        self.train_end = train_end
        self.valid_window = valid_window
        
        # 훈련 데이터와 검증 데이터 분리
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
        self.train_sequence_y = torch.tensor(self.train_sequence_y, dtype=torch.float32).unsqueeze(1)  # 1D로 변환

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

        # TensorDataset 생성
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
                # 모델 훈련
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

            # 검증 데이터로 모델 평가
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
                # 조기 종료 조건 확인
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    best_model = self.state_dict()
                    patience = 30
                else:
                    patience -= 1
                    if patience == 0:
                        print('Early stopping')
                        break
        # 최적 모델 저장
        torch.save(best_model, f'./{outputdir}/best_model.pth')
        # 모델 로드
        print('Best model loaded')