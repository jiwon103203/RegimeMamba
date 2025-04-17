import pandas as pd
import numpy as np
import torch
from jumpmodels.jump import JumpModel
from jumpmodels.preprocess import StandardScalerPD
from jumpmodels.plot import plot_regimes_and_cumret, savefig_plt

from .mamba_model import TimeSeriesMamba


class ModifiedJumpModel():
    """
    """

    def __init__(self, config, n_components=2, jump_penalty=50):
        """
        """

        super(ModifiedJumpModel, self).__init__()

        self.device = config.device
        self.seq_len = config.seq_len
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
        self.output_dir = config.results_dir
        self.jump_penalty = jump_penalty
        self.jm = JumpModel(n_components=n_components, jump_penalty=self.jump_penalty, cont=False)
        if config.input_dim == 5:
            self.feature_col = ['Open','Close','High','Low','treasury_rate']
        elif config.input_dim == 7:
            self.feature_col = ['Open','Close','High','Low','treasury_rate', 'treasury_rate_5y', 'dollar_index']
        self.scaler = StandardScalerPD()
    
    def train_for_window(self, train_start, train_end, data, sort = "cumret"):
        """
        """

        print(f"\nTraining period: {train_start} ~ {train_end}")

        train_data = data[(data['Date'] >= train_start) & (data['Date'] <= train_end)].copy()
        train_data['Open'] = train_data['Open'] / 100
        train_data['Close'] = train_data['Close'] / 100
        train_data['High'] = train_data['High'] / 100
        train_data['Low'] = train_data['Low'] / 100

        dates = train_data['Date'].values
        train_return_data = train_data.iloc[self.seq_len:] / 100
        train_data = train_data[self.feature_col]

        self.feature_extractor.eval()
        self.feature_extractor.to(self.device)
        hidden = []
        for i in range(0, len(train_data), self.seq_len):
            hidden.append(self.feature_extractor(torch.tensor(train_data.iloc[i:i+self.seq_len, :].values).to(self.device)).cpu().numpy()) # (16,)
        
        scaled_data = self.scaler.fit(np.array(hidden))
        self.jm.fit(scaled_data, train_return_data, sort_by=sort)

        ax, ax2 = plot_regimes_and_cumret(self.jm.labels_, train_return_data, n_c=2, start_date=train_start, end_date=train_end)
        ax.set(title=f"In-Sample Fitted Regimes by the JM(lambda : {self.jump_penalty})")
        savefig_plt(f"{self.output_dir}/JM_lambd_{self.jump_penalty}_train.pdf")

        return
    
    def predict(self, start_date, end_date, data, window_number, sort = "cumret"):
        """
        """

        print(f"\nPredicting period: {start_date} ~ {end_date}")

        pred_data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)].copy()
        pred_data['Open'] = pred_data['Open'] / 100
        pred_data['Close'] = pred_data['Close'] / 100
        pred_data['High'] = pred_data['High'] / 100
        pred_data['Low'] = pred_data['Low'] / 100

        # 초기 seq_len 개 데이터 무시
        pred_data = pred_data.iloc[self.seq_len:] / 100
        pred_data = pred_data[self.feature_col]

        self.feature_extractor.eval()
        self.feature_extractor.to(self.device)
        hidden = []
        for i in range(0, len(pred_data), self.seq_len):
            hidden.append(self.feature_extractor(torch.tensor(pred_data.iloc[i:i+self.seq_len, :].values).to(self.device)).cpu().numpy()) # (16,)
        
        scaled_data = self.scaler.transform(np.array(hidden))
        labels_test = self.jm.predict(scaled_data)

        ax, ax2 = plot_regimes_and_cumret(labels_test, pred_return_data, n_c=2, start_date=start_date, end_date=end_date)
        ax.set(title=f"Out-of-Sample Predicted Regimes by the JM(lambda : {self.jump_penalty})")
        savefig_plt(f"{self.output_dir}/JM_lambd_{self.jump_penalty}_test_window_{window_number}.pdf")
        