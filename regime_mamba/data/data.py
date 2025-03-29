import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import random

# 지수와 해당 국가의 3개월 국채 티커 정의
indices = {
    'SP500': {
        'index_ticker': '^GSPC',
        'treasury_ticker': '^IRX'  # 미국 3개월 국채
    }
    # },
    # 'Nikkei': {
    #     'index_ticker': '^N225',
    #     'treasury_ticker': '^JPTB3M'  # 일본 3개월 국채 (근사값)
    # },
    # 'DAX': {
    #     'index_ticker': '^GDAXI',
    #     'treasury_ticker': '^DE3M'  # 독일 3개월 국채 (근사값)
    # }
}

def download_data(ticker, start_date, end_date):
    """Yahoo Finance에서 데이터 다운로드"""
    try:
        data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
        if data.empty:
            print(f"{ticker}에 대한 데이터가 없습니다.")
            return None
        return data
    except Exception as e:
        print(f"{ticker} 다운로드 중 오류 발생: {e}")
        return None

def get_index_data(ticker, start_date, end_date, name):
    """지수 데이터 처리"""
    data = download_data(ticker, start_date, end_date)
    if data is None:
        return None

    # 수정 종가 사용
    df = pd.DataFrame(index=data.index)
    df[f'{name}_Close'] = data['Close']
    df[f'{name}_High'] = data['High']
    df[f'{name}_Low'] = data['Low']
    df[f'{name}_Open'] = data['Open']

    # 앞으로 N일 단순 이동평균(SMA)를 타겟으로 설정하기 - 20일, 60일, 120일
    df[f'{name}_target_SMA_20'] = data['Close'].rolling(window=20).mean().shift(-20)
    df[f'{name}_target_SMA_60'] = data['Close'].rolling(window=60).mean().shift(-60)
    df[f'{name}_target_SMA_120'] = data['Close'].rolling(window=120).mean().shift(-120)
    df[f'{name}_target_SMA_200'] = data['Close'].rolling(window=200).mean().shift(-200)

    # 로그 수익률 계산
    log_prices = np.log(data['Close'])
    df[f'{name}_returns'] = log_prices - log_prices.shift(1)
    df[f'{name}_target_returns_5'] = (log_prices - log_prices.shift(5)).shift(-5)
    df[f'{name}_target_returns_20'] = (log_prices - log_prices.shift(20)).shift(-20)
    df[f'{name}_target_returns_60'] = (log_prices - log_prices.shift(60)).shift(-60)
    df[f'{name}_target_returns_120'] = (log_prices - log_prices.shift(120)).shift(-120)
    df[f'{name}_target_returns_200'] = (log_prices - log_prices.shift(200)).shift(-200)
    df[f'{name}_returns_20'] = df[f'{name}_returns'].ewm(halflife=20, min_periods=1).mean().shift(1)
    df[f'{name}_returns_60'] = df[f'{name}_returns'].ewm(halflife=60, min_periods=1).mean().shift(1)

    return df

def get_treasury_data(ticker, start_date, end_date, name):
    """3개월 국채 데이터 처리"""
    data = download_data(ticker, start_date, end_date)
    if data is None:
        return None

    df = pd.DataFrame(index=data.index)
    df[f'{name}_treasury_rate'] = data['Close']

    # 일간 무위험 수익률 계산
    df[f'{name}_daily_rf'] = (1 + data['Close']/100)**(1/252) - 1
    df[f'{name}_daily_rf_20'] = df[f'{name}_daily_rf'].ewm(halflife=20, min_periods=1).mean().shift(1)
    df[f'{name}_daily_rf_60'] = df[f'{name}_daily_rf'].ewm(halflife=60, min_periods=1).mean().shift(1)

    return df

def calculate_metrics(data):
    """각 지수에 대한 지표 계산"""
    for name in indices.keys():
        # Downside Deviation 계산
        data[f'{name}_dd_10'] = np.sqrt(
            data[f'{name}_returns'].apply(lambda x: x**2 if x < 0 else 0)
            .ewm(halflife=10, min_periods=1).mean().shift(1)
        )
        data[f'{name}_dd_20'] = np.sqrt(
            data[f'{name}_returns'].apply(lambda x: x**2 if x < 0 else 0)
            .ewm(halflife=20, min_periods=1).mean().shift(1)
        )
        data[f'{name}_dd_60'] = np.sqrt(
            data[f'{name}_returns'].apply(lambda x: x**2 if x < 0 else 0)
            .ewm(halflife=60, min_periods=1).mean().shift(1)
        )

        # Sortino Ratio 계산
        data[f'{name}_sortino_20'] = (
            data[f'{name}_returns_20'] - data[f'{name}_daily_rf_20']
        ) / data[f'{name}_dd_20']

        data[f'{name}_sortino_60'] = (
            data[f'{name}_returns_60'] - data[f'{name}_daily_rf_60']
        ) / data[f'{name}_dd_60']

    return data

def preprocess_data():
    """전체 데이터 처리 및 병합"""
    # 충분한 과거 데이터 확보를 위해 일찍 시작
    start_date = "1955-01-01"
    end_date = "2024-12-31"

    # 모든 데이터프레임을 저장할 딕셔너리
    all_dfs = {}

    # 각 지수와 국채 데이터 다운로드 및 처리
    for name, tickers in indices.items():
        print(f"{name} 지수 데이터 처리 중...")
        index_df = get_index_data(tickers['index_ticker'], start_date, end_date, name)
        if index_df is not None:
            all_dfs[f'{name}_index'] = index_df

        time.sleep(random.randrange(60, 80))  # 60~80초 사이 대기

        print(f"{name} 국채 데이터 처리 중...")
        treasury_df = get_treasury_data(tickers['treasury_ticker'], start_date, end_date, name)
        if treasury_df is not None:
            all_dfs[f'{name}_treasury'] = treasury_df

        time.sleep(random.randrange(60, 80))  # 60~80초 사이 대기

    # 모든 데이터프레임 병합 (outer join)
    print("모든 데이터 병합 중...")
    merged_data = None

    for df_name, df in all_dfs.items():
        df.index.name = 'Date'
        if merged_data is None:
            merged_data = df
        else:
            # outer join으로 병합 (요구사항 3번)
            merged_data = merged_data.join(df, how='outer')

    # 데이터 날짜순 정렬
    merged_data = merged_data.sort_index()

    # NaN 값 채우기 (양 옆 값의 평균으로 채움)
    print("결측치 채우는 중...")
    merged_data = merged_data.interpolate(method='time')

    # 지표 계산
    merged_data = calculate_metrics(merged_data)

    # 데이터 범위 제한 (1960-01-02부터 2023-12-29까지)
    merged_data = merged_data.loc["1960-01-02":"2023-12-29"]

    # 혹시 남아있는 NaN 값 확인 및 제거
    if merged_data.isna().any().any():
        print("남아있는 NaN 값이 있습니다. 해당 행을 제거합니다.")
        merged_data = merged_data.dropna()

    return merged_data

data = preprocess_data()
# 전체 데이터 저장
data.to_csv("multi_index_data.csv")
print("데이터 처리 완료. 결과가 'multi_index_data.csv'에 저장되었습니다.")

# 주요 지표만 선택하여 별도 저장
selected_data = data[['SP500_Close', 'SP500_Open', 'SP500_High', 'SP500_Low', 'SP500_target_SMA_20', 'SP500_target_SMA_60', 'SP500_target_SMA_120', 'SP500_target_SMA_200', 'SP500_returns', 'SP500_target_returns_5', 'SP500_target_returns_20', 'SP500_target_returns_60', 'SP500_target_returns_120', 'SP500_target_returns_200', 'SP500_returns_20', 'SP500_returns_60', 'SP500_dd_10', 'SP500_dd_20', 'SP500_dd_60', 'SP500_sortino_20', 'SP500_sortino_60', 'SP500_treasury_rate', 'SP500_daily_rf', 'SP500_daily_rf_20', 'SP500_daily_rf_60']]
selected_data.to_csv('selected_metrics.csv')

for col in selected_data.columns:
  selected_data.rename(columns={col: col.replace('SP500_', '')}, inplace=True)


selected_data.to_csv('selected_metrics.csv')

