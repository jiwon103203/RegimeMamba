import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import torch
from .lstm import StackedLSTM
from jumpmodels.jump import JumpModel
from jumpmodels.preprocess import StandardScalerPD
from jumpmodels.plot import plot_regimes_and_cumret, savefig_plt

from .mamba_model import TimeSeriesMamba


class ModifiedJumpModel():
    """
    Modified Jump Model for Time Series Data
    This class implements a modified version of the Jump Model for time series data.
    It uses a feature extractor (TimeSeriesMamba) to extract features from the input data,
    and then applies the Jump Model to predict regimes based on the extracted features.
    The model is designed to work with time series data, and it can be trained and evaluated
    on different time windows.
    """

    def __init__(self, config, n_components=2, jump_penalty=32):
        """
        Initializes the ModifiedJumpModel with the given configuration.
        Args:
            config (object): Configuration object containing model parameters.
            n_components (int): Number of components for the Jump Model.
            jump_penalty (float): Penalty for jumps in the model.
        """
        super(ModifiedJumpModel, self).__init__()

        self.device = config.device
        self.seq_len = config.seq_len
        self.feature_extractor_type = 'lstm' if config.lstm else 'mamba'
        if config.lstm:
            self.feature_extractor = StackedLSTM(
                input_dim=config.input_dim,
            )
        else:
            self.feature_extractor = TimeSeriesMamba(
                input_dim=config.input_dim,
                d_model=config.d_model,
                d_state=config.d_state,
                n_layers=config.n_layers,
                dropout=config.dropout,
                config=config
            )

        self.vae = config.vae
        self.freeze_feature_extractor = config.freeze_feature_extractor
        if self.freeze_feature_extractor:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
                
        self.output_dir = config.results_dir
        self.jump_penalty = jump_penalty
        self.config = config
        self.jm = JumpModel(n_components=n_components, jump_penalty=self.jump_penalty, cont=False)
        self.original_jm = JumpModel(n_components=2, jump_penalty=50, cont=False)
        self.original_feature = ['dd_10', 'sortino_20', 'sortino_60']
        self.original_scaler = StandardScalerPD()

        # 차원에 따른 feature 컬럼 설정 간소화
        self.feature_col = self._get_feature_columns(config.input_dim)
        self.scaler = StandardScalerPD()
    
    def _get_feature_columns(self, input_dim):
        """차원에 따른 feature 컬럼 매핑"""
        feature_cols = {
            3: ['dd_10', 'sortino_20', 'sortino_60'],
            4: ['dd_10', 'sortino_20', 'sortino_60', 'dollar_index']
        }
        return feature_cols.get(input_dim, [])
    
    def _preprocess_data(self, data):
        """데이터 전처리 통합 메소드"""
        processed_data = data.copy()
        epsilon = 1e-10
        
        # OHLC 데이터 로그 변환
        if 'dollar_index' in processed_data.columns:
            processed_data['dollar_index'] = processed_data['dollar_index'] / self.config.scale
        # for col in ['Open', 'Close', 'High', 'Low']:
        #     if col in processed_data.columns:
        #         processed_data[col] = np.log(processed_data[col] + epsilon) - np.log(processed_data['Close'].shift(1) + epsilon)
        
        # NaN 값 처리
        processed_data = processed_data.fillna(0)
        
        return processed_data
    
    def _extract_features(self, data):
        """특성 추출 통합 메소드"""
        self.feature_extractor.eval()
        self.feature_extractor.to(self.device)
        hiddens = []
        
        # 원본 코드와 일치하도록 범위 수정 (+ 1 제거)
        if self.feature_extractor_type == 'lstm':
            for i in range(1, len(data)+1):
                with torch.no_grad():
                    if i < self.seq_len:
                        input_tensor = torch.tensor(
                            data.iloc[:i, :].values,
                            dtype=torch.float32
                        ).unsqueeze(0).to(self.device)
                    else:
                        input_tensor = torch.tensor(
                            data.iloc[i-self.seq_len:i, :].values, 
                            dtype=torch.float32
                        ).unsqueeze(0).to(self.device)
                    
                    _, hidden = self.feature_extractor(input_tensor, return_hidden=True)
                hiddens.append(hidden.squeeze().cpu().detach().numpy())
        else:            
            for i in range(1, len(data)+1):# - self.seq_len):
                with torch.no_grad():
                    if i < self.seq_len:
                        input_tensor = torch.tensor(
                            data.iloc[:i, :].values, 
                            dtype=torch.float32
                        ).unsqueeze(0).to(self.device)
                    else:
                        input_tensor = torch.tensor(
                            data.iloc[i-self.seq_len:i, :].values, 
                            dtype=torch.float32
                        ).unsqueeze(0).to(self.device)
                    
                    if self.vae:
                        _, _, _, _, hidden, _ = self.feature_extractor(input_tensor, return_hidden=True)
                    else:
                        _, hidden = self.feature_extractor(input_tensor, return_hidden=True)
            
                hiddens.append(hidden.squeeze().cpu().detach().numpy())
        
        return np.stack(hiddens)
    
    def _parse_date(self, date_str):
        """날짜 문자열 파싱 통합 메소드"""
        if ' ' in date_str:
            date_str = date_str.split(' ')[0]
        return datetime.strptime(date_str, "%Y-%m-%d")
    
    def _adjust_date(self, date_str, days=None):
        """시작 날짜 조정 통합 메소드"""
        days = days if days is not None else self.seq_len - 1
        date = self._parse_date(date_str)
        return (date + relativedelta(days=days)).strftime("%Y-%m-%d")
    
    def train_for_window(self, train_start, train_end, data, valid_window, sort="cumret", window=1, dynamic=False, penalty_range=None):
        """
        Trains the model on a specified time window with optional dynamic jump penalty selection.
        Args:
            train_start (str): Start date for training data in 'YYYY-MM-DD' format.
            train_end (str): End date for training data in 'YYYY-MM-DD' format.
            data (pd.DataFrame): Input data containing time series features and returns.
            valid_window (float): Validation window size in years.
            sort (str): Sorting criterion for the Jump Model ('cumret' or 'other').
            window (int): Window number for training.
            dynamic (bool): Whether to dynamically select the best jump penalty.
            penalty_range (list): List of jump penalties to evaluate. If None, uses default range.
        Returns:
            None
        """
        # 날짜 문자열 정리 및 확장된 훈련용 종료 날짜 계산
        if self.config.train_years > 16:
            train_end_clean = train_end.split(" ")[0]
            train_start_clean = (self._parse_date(train_end_clean) - relativedelta(years = 20)).strftime("%Y-%m-%d") 
        else:
            train_start_clean = train_start.split(" ")[0]
            train_end_clean = train_end.split(" ")[0]
        
        print(f"\nTraining period: {train_start_clean} ~ {train_end_clean}")
        
        # 원본 JumpModel 훈련
        original_end_date = (self._parse_date(train_end_clean) + relativedelta(years=valid_window)).strftime("%Y-%m-%d")
        original_train_data = data[(data['Date'] >= train_start_clean) & (data['Date'] <= original_end_date)].copy()
        original_features = original_train_data[self.original_feature].copy()
        original_train_return_data = original_train_data['returns']
        
        # 일관된 인덱스 설정
        common_dates = pd.to_datetime(original_train_data['Date'].values)
        original_train_return_data.index = common_dates
        original_features.index = common_dates
        
        # 원본 JumpModel 피팅
        self.original_scaler.fit(original_features)
        original_scaled_data = self.original_scaler.transform(original_features)
        self.original_jm.fit(original_scaled_data, original_train_return_data, sort_by=sort)
        
        # 원본 JumpModel 결과 시각화
        ax, ax2 = plot_regimes_and_cumret(
            self.original_jm.labels_, 
            original_train_return_data, 
            n_c=2, 
            start_date=train_start_clean, 
            end_date=train_end_clean
        )
        ax.set(title=f"In-Sample Fitted Regimes by the Original JM(lambda : 50)")
        savefig_plt(f"{self.output_dir}/window_{window}/JM_lambd_50_train.png")
        
        # 수정된 JumpModel을 위한 데이터 전처리
        train_data = self._preprocess_data(
            data[(data['Date'] >= train_start_clean) & (data['Date'] <= train_end)]
        )
        
        # 시퀀스 길이를 고려한 시작 날짜 조정
        adjusted_start_date = train_start_clean
        
        # 수정된 JumpModel을 위한 데이터 준비
        common_index = pd.to_datetime(train_data['Date']).iloc[1:]
        
        # 일관된 인덱스로 리턴 데이터 준비
        train_return_data = train_data['returns'].iloc[1:]
        train_return_data.index = common_index
        
        # 특성 추출
        feature_data = train_data[self.feature_col]
        hiddens = []
        for i in range(1, len(feature_data)):
            with torch.no_grad():
                if i < self.seq_len:
                    input_tensor = torch.tensor(
                        feature_data.iloc[:i, :].values,
                        dtype=torch.float32
                    ).unsqueeze(0).to(self.device)
                else:
                    input_tensor = torch.tensor(
                        feature_data.iloc[i-self.seq_len:i, :].values, 
                        dtype=torch.float32
                    ).unsqueeze(0).to(self.device)
                
                if self.vae:
                    _, _, _, _, hidden, _ = self.feature_extractor(input_tensor, return_hidden=True)
                else:
                    _, hidden = self.feature_extractor(input_tensor, return_hidden=True)
            
            hiddens.append(hidden.squeeze().cpu().detach().numpy())
        
        hiddens = np.stack(hiddens)
        
        # 적절한 인덱스와 함께 DataFrame으로 변환
        hiddens_df = pd.DataFrame(
            hiddens, 
            columns=[f"hidden_{i}" for i in range(hiddens.shape[1])],
            index=common_index
        )
        
        # 스케일러 학습
        self.scaler.fit(hiddens_df)
        scaled_data = self.scaler.transform(hiddens_df)
        
        # 동적 점프 페널티 선택 로직 (dynamic=True인 경우)
        if dynamic:
            # 검증 데이터 준비
            valid_start = train_end_clean
            valid_end = (self._parse_date(train_end_clean) + relativedelta(years=valid_window)).strftime("%Y-%m-%d")
            
            print(f"Validation period for dynamic penalty selection: {valid_start} ~ {valid_end}")
            
            # 검증 데이터 전처리
            valid_data = self._preprocess_data(
                data[(data['Date'] > valid_start) & (data['Date'] <= valid_end)].copy()
            )
            
            if len(valid_data) == 0:
                print("Warning: No validation data available. Using default jump penalty.")
            else:
                # 검증 데이터에서 특성 추출
                valid_feature_data = valid_data[self.feature_col]
                valid_hiddens = []
                
                for i in range(1, len(valid_feature_data)):
                    with torch.no_grad():
                        if i < self.seq_len:
                            input_tensor = torch.tensor(
                                valid_feature_data.iloc[:i, :].values,
                                dtype=torch.float32
                            ).unsqueeze(0).to(self.device)
                        else:
                            input_tensor = torch.tensor(
                                valid_feature_data.iloc[i-self.seq_len:i, :].values, 
                                dtype=torch.float32
                            ).unsqueeze(0).to(self.device)
                        
                        if self.vae:
                            _, _, _, _, hidden, _ = self.feature_extractor(input_tensor, return_hidden=True)
                        else:
                            _, hidden = self.feature_extractor(input_tensor, return_hidden=True)
                    
                    valid_hiddens.append(hidden.squeeze().cpu().detach().numpy())
                
                valid_hiddens = np.stack(valid_hiddens)
                
                # 검증 데이터 인덱스 설정
                valid_dates = valid_data['Date'].iloc[1:]
                valid_index = pd.to_datetime(valid_dates)
                
                # 검증 데이터 리턴
                valid_return_data = valid_data['returns'].iloc[1:]
                valid_return_data.index = valid_index
                
                # 검증 데이터 특성 DataFrame 생성
                valid_hiddens_df = pd.DataFrame(
                    valid_hiddens, 
                    columns=[f"hidden_{i}" for i in range(valid_hiddens.shape[1])],
                    index=valid_index
                )
                
                # 검증 데이터 스케일링
                valid_scaled_data = self.scaler.transform(valid_hiddens_df)
                
                # 페널티 범위 설정 (기본값 또는 사용자 지정)
                if penalty_range is None:
                    penalty_range = [16, 25, 32, 50, 64]
                
                # 각 페널티 값에 대한 검증 성능 평가
                best_penalty = self.jump_penalty  # 기본값
                best_performance = float('-inf')  # 초기화 (가장 나쁜 성능)
                
                # 각 페널티에 대한 결과 저장을 위한 딕셔너리
                penalty_results = {}
                
                print("Evaluating jump penalties on validation data...")
                
                for penalty in penalty_range:
                    # 해당 페널티로 JumpModel 생성 및 학습
                    test_jm = JumpModel(n_components=self.jm.n_components, jump_penalty=penalty, cont=False)
                    test_jm.fit(scaled_data, train_return_data, sort_by=sort)
                    
                    # 검증 데이터로 예측
                    test_labels = test_jm.predict_online(valid_scaled_data)
                    test_labels = pd.Series(test_labels, index=valid_index).fillna(0).astype(int)
                    
                    # 성능 평가 (누적 수익률로 평가)
                    # 각 레이블에 따른 리턴 분리
                    regime_returns = {}
                    for label in test_labels.unique():
                        regime_mask = test_labels == label
                        regime_returns[label] = valid_return_data[regime_mask]
                    
                    # 각 레짐의 누적 수익률 계산
                    cumulative_returns = {}
                    for label, returns in regime_returns.items():
                        if len(returns) > 0:
                            cumulative_returns[label] = (1 + returns).cumprod().iloc[-1] - 1 if len(returns) > 0 else 0
                        else:
                            cumulative_returns[label] = 0
                    
                    # 가장 좋은 레짐과 가장 나쁜 레짐의 수익률 차이로 성능 측정
                    if len(cumulative_returns) >= 2:
                        returns_list = list(cumulative_returns.values())
                        performance = max(returns_list) - min(returns_list)
                    else:
                        performance = 0
                    
                    # 결과 저장
                    penalty_results[penalty] = {
                        'performance': performance,
                        'cumulative_returns': cumulative_returns,
                        'model': test_jm
                    }
                    
                    print(f"Jump penalty {penalty}: Performance = {performance:.4f}, Cumulative returns = {cumulative_returns}")
                    
                    # 최고 성능 모델 업데이트
                    if performance > best_performance:
                        best_performance = performance
                        best_penalty = penalty
                
                # 최적의 페널티 출력 및 설정
                print(f"Selected best jump penalty: {best_penalty} with performance: {best_performance:.4f}")
                
                # 최적의 JumpModel 선택
                self.jump_penalty = best_penalty
                self.jm = penalty_results[best_penalty]['model']
                
                # 검증 결과 시각화
                import matplotlib.pyplot as plt
                
                penalties = list(penalty_results.keys())
                performances = [penalty_results[p]['performance'] for p in penalties]
                
                plt.figure(figsize=(10, 6))
                plt.plot(penalties, performances, 'o-')
                plt.axvline(x=best_penalty, color='r', linestyle='--', label=f'Best penalty: {best_penalty}')
                plt.xlabel('Jump Penalty')
                plt.ylabel('Performance (Regime Return Difference)')
                plt.title('Jump Penalty Selection Performance')
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                savefig_plt(f"{self.output_dir}/window_{window}/jump_penalty_selection.png")
        else:
            # 동적 선택을 사용하지 않는 경우 기존 로직 유지
            self.jm.fit(scaled_data, train_return_data, sort_by=sort)
        
        # 수정된 JumpModel 결과 시각화
        ax, ax2 = plot_regimes_and_cumret(
            self.jm.labels_, 
            train_return_data, 
            n_c=2, 
            start_date=adjusted_start_date, 
            end_date=train_end_clean
        )
        ax.set(title=f"In-Sample Fitted Regimes by the JM(lambda : {self.jump_penalty})")
        savefig_plt(f"{self.output_dir}/window_{window}/JM_lambd_{self.jump_penalty}_train.png")

    def predict(self, start_date, end_date, data, window_number, sort="cumret"):
        """
        Predicts regimes for a specified time window.
        Args:
            start_date (str): Start date for prediction data in 'YYYY-MM-DD' format.
            end_date (str): End date for prediction data in 'YYYY-MM-DD' format.
            data (pd.DataFrame): Input data containing time series features and returns.
            window_number (int): Window number for prediction.
            sort (str): Sorting criterion for the Jump Model ('cumret' or 'other').
        Returns:
            None
        """
        print(f"\nPredicting period: {start_date} ~ {end_date}")
        
        # 데이터 전처리
        pred_data = self._preprocess_data(
            data[(data['Date'] >= start_date) & (data['Date'] <= end_date)].copy()
        )
        
        # 시퀀스 길이를 고려한 날짜 조정
        adjusted_start_date = pd.to_datetime(start_date)#pd.to_datetime(self._adjust_date(start_date))
        dates = pred_data['Date'].iloc[1:]#.iloc[self.seq_len-1:]
        common_index = pd.to_datetime(dates)
        
        # 리턴 데이터 준비
        pred_return_data = pred_data['returns'].iloc[1:]#.iloc[self.seq_len-1:]
        pred_return_data.index = common_index
        
        # 원본 모델 예측
        original_features = pred_data[self.original_feature]#.iloc[self.seq_len-1:]
        original_labels_test = self.original_jm.predict_online(
            self.original_scaler.transform(original_features)
        )

        pd.DataFrame({
            'Date': dates.values,
            'Label': original_labels_test.values[1:]
        }).to_csv(f"{self.output_dir}/window_{window_number}/original_labels.csv", index=False)
        
        
        # 적절한 인덱스로 라벨 변환 및 NaN 처리
        original_labels_test = pd.Series(original_labels_test, index=common_index).fillna(0).astype(int)
        
        # 원본 모델 결과 시각화
        ax, ax2 = plot_regimes_and_cumret(
            original_labels_test, 
            pred_return_data, 
            n_c=2, 
            start_date=adjusted_start_date, 
            end_date=end_date
        )
        ax.set(title=f"Out-of-Sample Predicted Regimes by the Original JM(lambda : 50)")
        savefig_plt(f"{self.output_dir}/window_{window_number}/JM_lambd_50_test_window.png")
        
        # 원본 모델 결과 저장
        df_original = pred_data.copy()
        df_original['labels'] = -1
        df_original['labels'] = original_labels_test #.iloc[self.seq_len-1:] = original_labels_test
        df_original['Date'] = pd.to_datetime(dates)
        df_original = df_original[['Date', 'labels']]
        df_original.to_csv(f"{self.output_dir}/window_{window_number}/JM_lambd_50_test_window.csv", index=False)
        
        # 수정된 모델용 특성 추출
        feature_data = pred_data[self.feature_col]
        hiddens = []
         # 원본 코드와 정확히 동일한 범위 사용
        for i in range(1, len(feature_data)):# - self.seq_len + 1):
            with torch.no_grad():
                if i < self.seq_len:
                    input_tensor = torch.tensor(
                        feature_data.iloc[:i, :].values, 
                        dtype=torch.float32
                    ).unsqueeze(0).to(self.device)
                else:
                    input_tensor = torch.tensor(
                        feature_data.iloc[i-self.seq_len:i, :].values, 
                        dtype=torch.float32
                    ).unsqueeze(0).to(self.device)
                
                if self.vae:
                    _, _, _, _, hidden, _ = self.feature_extractor(input_tensor, return_hidden=True)
                else:
                    _, hidden = self.feature_extractor(input_tensor, return_hidden=True)
            
            hiddens.append(hidden.squeeze().cpu().detach().numpy())
        
        hiddens = np.stack(hiddens)
        
        # DataFrame으로 변환
        hiddens_df = pd.DataFrame(
            hiddens, 
            columns=[f"hidden_{i}" for i in range(hiddens.shape[1])],
            index=common_index
        )
        
        # 수정된 모델 예측
        scaled_data = self.scaler.transform(hiddens_df)
        labels_test = self.jm.predict_online(scaled_data)
        pd.DataFrame({
            'Date': dates.values,
            'Label': labels_test
        }).to_csv(f"{self.output_dir}/window_{window_number}/labels.csv", index=False)
        
        # 적절한 인덱스로 라벨 변환 및 NaN 처리
        labels_test = pd.Series(labels_test, index=common_index).fillna(0).astype(int)
        
        # 수정된 모델 결과 시각화
        ax, ax2 = plot_regimes_and_cumret(
            labels_test, 
            pred_return_data, 
            n_c=2, 
            start_date=adjusted_start_date, 
            end_date=end_date
        )
        ax.set(title=f"Out-of-Sample Predicted Regimes by the JM(lambda : {self.jump_penalty})")
        savefig_plt(f"{self.output_dir}/window_{window_number}/JM_lambd_{self.jump_penalty}_test_window.png")
        
        df_modified = pred_data.copy()
        df_modified['labels'] = -1
        df_modified['labels'] = labels_test #.iloc[self.seq_len-1:] = labels_test
        df_modified['Date'] = pd.to_datetime(dates)
        df_modified = df_modified[['Date', 'labels']]
        df_modified.to_csv(f"{self.output_dir}/window_{window_number}/JM_lambd_{self.jump_penalty}_test_window.csv", index=False)