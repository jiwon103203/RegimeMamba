import torch
import os

class RegimeMambaConfig:
    def __init__(self):
        """RegimeMamba 모델 설정 클래스"""
        # 데이터 관련 설정
        self.data_path = None
        self.target_type = "next_day"
        self.target_horizon = 1
        self.preprocessed = False
        
        # 모델 구조 관련 설정
        self.d_model = 128
        self.d_state = 128
        self.d_conv = 4
        self.expand = 2
        self.n_layers = 4
        self.dropout = 0.1
        self.input_dim = 4
        self.seq_len = 128
        
        # 훈련 관련 설정
        self.batch_size = 64
        self.learning_rate = 1e-6
        self.max_epochs = 50
        self.patience = 10
        self.transaction_cost = 0.001
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.direct_train = False
        self.vae = False
        self.use_onecycle = True
        self.progressive_train = 0

        # 클러스터링 관련 설정
        self.n_clusters = 2  # Bull과 Bear 두 개의 레짐으로 클러스터링
        self.cluster_method = 'cosine_kmeans'

        # Extra 설정
        self.jump_model = False
        self.rl_model = False
        self.rl_learning_rate = 1e-4
        self.rl_gamma = 0.99
        self.freeze_feature_extractor = True
        self.position_penalty = 0.01
        self.reward_type = 'sharpe'
        self.window_size = 252
        self.n_episodes = 50
        self.steps_per_episode = 2048
        self.n_epochs = 50
        self.rl_batch_size = 512
        self.n_positions = 3
        self.optimize_thresholds = False

        # 예측 관련 설정
        self.predict = False

    def __str__(self):
        """설정 정보를 문자열로 반환"""
        config_str = "RegimeMamba 설정:\n"
        for key, value in self.__dict__.items():
            config_str += f"  {key}: {value}\n"
        return config_str
        
    def save(self, filepath):
        """설정을 JSON 파일로 저장"""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.__dict__, f, indent=4, default=str)
        
    @classmethod
    def load(cls, filepath):
        """JSON 파일에서 설정 로드"""
        import json
        config = cls()
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
            for key, value in config_dict.items():
                setattr(config, key, value)
        
        # device 객체 복원
        config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return config

class RollingWindowConfig(RegimeMambaConfig):

    def __init__(self):
        """Rolling Window 기반 RegimeMamba 모델 설정 클래스"""
        super().__init__()
        self.lookback_years = 10      # 클러스터링에 사용할 과거 데이터 기간(년)
        self.forward_months = 12      # 적용할 미래 기간(개월)
        self.start_date = '2010-01-01'  # 백테스트 시작일
        self.end_date = '2023-12-31'    # 백테스트 종료일
        self.transaction_cost = 0.001 # 거래 비용 (0.1%)
        self.model_path = None        # 사전 훈련된 모델 경로

        # 저장 경로
        self.results_dir = './rolling_window_results'
        os.makedirs(self.results_dir, exist_ok=True)

    def __str__(self):
        """설정 정보를 문자열로 반환"""
        config_str = "RollingWindowConfig 설정:\n"
        for key, value in self.__dict__.items():
            config_str += f"  {key}: {value}\n"
        return config_str
    
    @classmethod
    def load(cls, filepath):
        """JSON 파일에서 설정 로드"""
        import json
        config = cls()
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
            for key, value in config_dict.items():
                setattr(config, key, value)
        
        # device 객체 복원
        config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return config
    
class RollingWindowTrainConfig(RollingWindowConfig):
    def __init__(self):
        """Rolling Window 기반 RegimeMamba 훈련 설정 클래스"""
        super().__init__()
        self.total_window_years = 40
        self.train_years = 20
        self.valid_years = 10
        self.clustering_years = 10
        self.forward_months = 60

        # 훈련 관련 설정
        self.max_epochs = 100

        self.apply_filtering = True
        self.filter_method = 'minimum_holding'
        self.min_holding_days = 20

        self.results_dir = './rolling_window_train_results'
        os.makedirs(self.results_dir, exist_ok=True)

    def __str__(self):
        """설정 정보를 문자열로 반환"""
        config_str = "RollingWindowTrainconfig 설정:\n"
        for key, value in self.__dict__.items():
            config_str += f"  {key}: {value}\n"
        return config_str
    
    @classmethod
    def load(cls, filepath):
        """JSON 파일에서 설정 로드"""
        import json
        config = cls()
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
            for key, value in config_dict.items():
                setattr(config, key, value)
        
        # device 객체 복원
        config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return config
