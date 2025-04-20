import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import torch
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

    Attributes:
        config (object): Configuration object containing model parameters.
        n_components (int): Number of components for the Jump Model.
        jump_penalty (float): Penalty for jumps in the model.
        feature_extractor (TimeSeriesMamba): Feature extractor for time series data.
        vae (bool): Flag indicating whether to use a Variational Autoencoder.
        freeze_feature_extractor (bool): Flag indicating whether to freeze the feature extractor parameters.
        output_dir (str): Directory to save output results.
        scaler (StandardScalerPD): Standard scaler for preprocessing the data.
        jm (JumpModel): Instance of the Jump Model.
        feature_col (list): List of feature columns to be used in the model.
    Methods:
        __init__(config, n_components=2, jump_penalty=1.5): Initializes the ModifiedJumpModel with the given configuration.
        train_for_window(train_start, train_end, data, sort="cumret", window=1): Trains the model on a specified time window.
        predict(start_date, end_date, data, window_number, sort="cumret"): Predicts regimes for a specified time window.
    Example:
        config = Config()  # Assuming you have a configuration object
        model = ModifiedJumpModel(config, n_components=2, jump_penalty=1.5)
        model.train_for_window("2020-01-01", "2020-12-31", data)
        model.predict("2021-01-01", "2021-12-31", data, window_number=1)
    """

    def __init__(self, config, n_components=2, jump_penalty=1.5):
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
        self.jm = JumpModel(n_components=n_components, jump_penalty=self.jump_penalty, cont=False)
        if config.input_dim == 5:
            self.feature_col = ['Open','Close','High','Low','treasury_rate']
        elif config.input_dim == 7:
            self.feature_col = ['Open','Close','High','Low','treasury_rate', 'treasury_rate_5y', 'dollar_index']
        elif config.input_dim == 8:
            self.feature_col = ["Open", "Close", "High", "Low", "Volume","treasury_rate", "treasury_rate_5y", "dollar_index"]
        self.scaler = StandardScalerPD()
    
    def train_for_window(self, train_start, train_end, data, sort = "cumret", window = 1):
        """
        Trains the model on a specified time window.
        Args:
            train_start (str): Start date for training data in 'YYYY-MM-DD' format.
            train_end (str): End date for training data in 'YYYY-MM-DD' format.
            data (pd.DataFrame): Input data containing time series features and returns.
            sort (str): Sorting criterion for the Jump Model ('cumret' or 'other').
            window (int): Window number for training.
        Returns:
            None
        """

        print(f"\nTraining period: {train_start} ~ {train_end}")

        train_data = data[(data['Date'] >= train_start) & (data['Date'] <= train_end)].copy()
        train_data['Open'] = train_data['Open'] / 100
        train_data['Close'] = train_data['Close'] / 100
        train_data['High'] = train_data['High'] / 100
        train_data['Low'] = train_data['Low'] / 100
        train_start = str(datetime.strptime(train_start, "%Y-%m-%d") + relativedelta(days=self.seq_len - 1))


        dates = train_data['Date'].values
        common_index = pd.to_datetime(dates[self.seq_len:])
        train_return_data = train_data['returns'].iloc[self.seq_len:]
        train_return_data.index = common_index
        train_data = train_data[self.feature_col]

        self.feature_extractor.eval()
        self.feature_extractor.to(self.device)
        hiddens = []
        for i in range(0, len(train_data) - self.seq_len):
            if self.vae:
                _, _, _, _, hidden, _ = self.feature_extractor(torch.tensor(train_data.iloc[i:i+self.seq_len, :].values, dtype=torch.float32).unsqueeze(0).to(self.device), return_hidden = True)
            else:
                _, hidden = self.feature_extractor(torch.tensor(train_data.iloc[i:i+self.seq_len, :].values, dtype=torch.float32).unsqueeze(0).to(self.device), return_hidden = True) # (16,)

            hiddens.append(hidden.squeeze().cpu().detach().numpy())
        
        hiddens = np.stack(hiddens)
        hiddens = pd.DataFrame(hiddens, columns=[f"hidden_{i}" for i in range(hiddens.shape[1])])
        hiddens.index = common_index
        self.scaler.fit(hiddens)
        scaled_data = self.scaler.transform(hiddens)
        self.jm.fit(scaled_data, train_return_data, sort_by=sort)

        ax, ax2 = plot_regimes_and_cumret(self.jm.labels_, train_return_data, n_c=2, start_date=train_start, end_date=train_end)
        ax.set(title=f"In-Sample Fitted Regimes by the JM(lambda : {self.jump_penalty})")
        savefig_plt(f"{self.output_dir}/JM_lambd_{self.jump_penalty}_train_{window}.png")

        return
    
    def predict(self, start_date, end_date, data, window_number, sort = "cumret"):
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
        
        pred_data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)].copy()
        pred_data['Open'] = pred_data['Open'] / 100
        pred_data['Close'] = pred_data['Close'] / 100
        pred_data['High'] = pred_data['High'] / 100
        pred_data['Low'] = pred_data['Low'] / 100
        start_date = str(datetime.strptime(start_date, "%Y-%m-%d") + relativedelta(days=self.seq_len - 1))


        # 초기 seq_len 개 데이터 무시
        dates = pred_data['Date'].values
        common_index = pd.to_datetime(dates[self.seq_len:])
        pred_return_data = pred_data['returns'].iloc[self.seq_len:]
        pred_return_data.index = common_index
        pred_data = pred_data[self.feature_col]

        self.feature_extractor.eval()
        self.feature_extractor.to(self.device)
        hiddens = []
        for i in range(0, len(pred_data)-self.seq_len):
            if self.vae:
                _, _, _, _, hidden, _ = self.feature_extractor(torch.tensor(pred_data.iloc[i:i+self.seq_len, :].values, dtype=torch.float32).unsqueeze(0).to(self.device), return_hidden = True)
            else:
                _, hidden = self.feature_extractor(torch.tensor(pred_data.iloc[i:i+self.seq_len, :].values, dtype=torch.float32).unsqueeze(0).to(self.device), return_hidden = True) # (16,)
            hiddens.append(hidden.squeeze().cpu().detach().numpy())
        
        hiddens = np.stack(hiddens)
        hiddens = pd.DataFrame(hiddens, columns=[f"hidden_{i}" for i in range(hiddens.shape[1])])
        hiddens.index = common_index
        scaled_data = self.scaler.transform(hiddens)
        labels_test = self.jm.predict(scaled_data)

        ax, ax2 = plot_regimes_and_cumret(labels_test, pred_return_data, n_c=2, start_date=start_date, end_date=end_date)
        ax.set(title=f"Out-of-Sample Predicted Regimes by the JM(lambda : {self.jump_penalty})")
        savefig_plt(f"{self.output_dir}/JM_lambd_{self.jump_penalty}_test_window_{window_number}.png")
        