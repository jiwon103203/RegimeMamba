import torch

class RegimeMambaConfig:
    def __init__(self):
        """RegimeMamba 모델 설정 클래스"""
        # 데이터 관련 설정
        self.data_path = None
        self.seq_len = 128
        
        # 모델 구조 관련 설정
        self.d_model = 128
        self.d_state = 128
        self.d_conv = 4
        self.expand = 2
        self.n_layers = 4
        self.dropout = 0.1
        
        # 훈련 관련 설정
        self.batch_size = 64
        self.learning_rate = 1e-6
        self.epochs = 1000
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 클러스터링 관련 설정
        self.n_clusters = 2  # Bull과 Bear 두 개의 레짐으로 클러스터링

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