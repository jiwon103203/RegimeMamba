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
from torch.utils.data import DataLoader, Dataset

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
        input_size (int): 입력 특성의 차원
        hidden_size (int): LSTM의 은닉 상태 차원
        num_layers (int): LSTM 레이어의 수
        output_size (int): 출력 특성의 차원
    '''
    def __init__(self, input_size, hidden_size=32, num_layers=8, output_size=1):
        super(StackedLSTM, self).__init__()
        
        # LSTM 레이어 초기화
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # 드롭아웃 레이어 초기화
        self.dropout = nn.Dropout(0.1)
        
        # 출력층 초기화
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # LSTM 레이어 통과
        lstm_out, _ = self.lstm(x)
        
        # 마지막 시퀀스 출력 선택
        lstm_out = lstm_out[:, -1, :]
        
        # 드롭아웃 적용
        out = self.dropout(lstm_out)
        
        # 출력층 통과
        out = self.fc(out)
        
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
        self.train_data = data[train_start:train_end]
        # train_end: str -> datetime
        self.valid_end = str(datetime.strptime(train_end, '%Y-%m-%d') + relativedelta(years=valid_window))
        self.valid_data = data[train_end:self.valid_end]

        train_dataloader = DataLoader(self.train_data, batch_size=2048, shuffle=False)
        valid_dataloader = DataLoader(self.valid_data, batch_size=2048, shuffle=False)

        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        patience = 30
        best_loss = float('inf')
        best_model = None

        for epoch in range(1000):
            for i, (x, y) in enumerate(train_dataloader):
                # 모델 훈련
                self.train()
                optimizer.zero_grad()
                output = self(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                
                if i % 100 == 0:
                    print(f'Epoch [{epoch+1}/200], Step [{i+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}')

            # 검증 데이터로 모델 평가
            self.eval()
            valid_loss = 0
            with torch.no_grad():
                for x, y in valid_dataloader:
                    output = self(x)
                    loss = criterion(output, y)
                    valid_loss += loss.item()
                valid_loss /= len(valid_dataloader)
                print(f'Validation Loss: {valid_loss:.4f}')
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