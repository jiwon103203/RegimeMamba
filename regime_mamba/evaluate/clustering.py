import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

def extract_hidden_states(model, dataloader, config):
    """
    모델로부터 hidden states와 타겟 수익률을 추출

    Args:
        model: 평가할 모델
        dataloader: 데이터 로더
        config: 설정 객체

    Returns:
        hidden_states: 추출된 hidden states
        returns: 타겟 수익률
        dates: 날짜 정보
    """
    model.eval()
    hidden_states = []
    returns = []
    dates = []

    with torch.no_grad():
        if config.preprocessed:
            for x, y, date in dataloader:
                x = x.to(config.device)
                if config.vae:
                    _, _, _, _, hidden, _ = model(x)
                else:
                    _, hidden = model(x, return_hidden=True)
                hidden_states.append(hidden.cpu().numpy())
                returns.append(y.numpy().reshape(-1,1))
                dates.extend(date)
        else:
            for x, y, date, r in dataloader:
                x = x.to(config.device)
                if config.vae:
                    _, _, _, _, hidden, _ = model(x)
                else:
                    _, hidden = model(x, return_hidden=True)
                hidden_states.append(hidden.cpu().numpy())
                returns.append(r.numpy().reshape(-1,1))
                dates.extend(date)

    hidden_states = np.vstack(hidden_states) #  행 단위로 쌓기 (n_samples, hidden_size)
    returns = np.vstack(returns) # 행 단위로 쌓기 (n_samples, 1)


    return hidden_states, returns, dates

def cosine_kmeans(hidden_states, n_clusters=2, random_state=42, max_iter=300):
    """
    코사인 유사도 기반 KMeans 클러스터링 구현
    
    Args:
        hidden_states: 클러스터링할 hidden states (n_samples, hidden_size)
        n_clusters: 클러스터 수
        random_state: 랜덤 시드
        max_iter: 최대 반복 횟수
        
    Returns:
        kmeans: 훈련된 KMeans 모델
        normalized_states: 정규화된 hidden states
        clusters: 클러스터 할당 레이블
    """
    # 1. 데이터 정규화 (L2 norm): 코사인 유사도를 위한 필수 단계
    normalized_states = normalize(hidden_states, norm='l2')
    
    # 2. 표준 KMeans 적용 (정규화된 데이터에 유클리드 거리를 사용하면 코사인 유사도와 동일)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, max_iter=max_iter)
    clusters = kmeans.fit_predict(normalized_states)
    
    return kmeans, normalized_states, clusters

def identify_bull_bear_regimes(hidden_states, returns, config):
    """
    hidden states를 클러스터링하고 Bull/Bear 레짐 식별

    Args:
        hidden_states: 추출된 hidden states
        returns: 타겟 수익률
        config: 설정 객체

    Returns:
        kmeans: 훈련된 KMeans 모델
        bull_regime: Bull 레짐 클러스터 ID
    """
    if config.cluster_method == 'cosine_kmeans':
        kmeans, hidden_states, clusters = cosine_kmeans(hidden_states, n_clusters=config.n_clusters, random_state=42)
    
    elif config.cluster_method == 'kmeans':
        # K-Means 클러스터링
        kmeans = KMeans(n_clusters=config.n_clusters, random_state=42)
        clusters = kmeans.fit_predict(hidden_states)
    
    # 실루엣 점수 계산 (클러스터링 품질 평가)
    try:
        silhouette_avg = silhouette_score(hidden_states, clusters)
        print(f"실루엣 점수: {silhouette_avg:.3f}")
    except:
        print("실루엣 점수 계산 실패")

    # 각 클러스터의 평균 수익률 계산
    cluster_returns = {}
    for i in range(config.n_clusters):
        cluster_mask = (clusters == i)
        avg_return = returns[cluster_mask].mean()
        cluster_returns[i] = avg_return

    # 가장 높은 평균 수익률을 가진 클러스터를 Bull 레짐으로 선택
    bull_regime = max(cluster_returns, key=cluster_returns.get)
    if config.n_clusters == 3:
        
        bear_regime = min(cluster_returns, key=cluster_returns.get)

        print(f"클러스터 평균 수익률: {cluster_returns}")
        print(f"Bull 레짐 클러스터: {bull_regime}")
        print(f"Bear 레짐 클러스터: {bear_regime}")

        return kmeans, bull_regime, bear_regime

    print(f"클러스터 평균 수익률: {cluster_returns}")
    print(f"Bull 레짐 클러스터: {bull_regime}")

    return kmeans, bull_regime

def predict_regimes(model, dataloader, kmeans, bull_regime, config, bear_regime=None):
    """
    테스트 데이터에 대해 레짐 예측

    Args:
        model: 평가할 모델
        dataloader: 데이터 로더
        kmeans: 훈련된 KMeans 모델
        bull_regime: Bull 레짐 클러스터 ID
        config: 설정 객체

    Returns:
        predictions: 예측된 레짐 (1=Bull, 0=Bear)
        true_returns: 실제 수익률
        dates: 날짜 정보
    """
    model.eval()
    predictions = []
    true_returns = []
    dates = []

    with torch.no_grad():
        if config.preprocessed:
            for x, y, date in dataloader:
                x = x.to(config.device)
                _, hidden = model(x, return_hidden=True)
                hidden = hidden.cpu().numpy()

                # 클러스터 할당
                cluster = kmeans.predict(hidden)

                if config.n_clusters == 2:
                    # Bull 레짐이면 1, 아니면 0
                    regime_pred = np.where(cluster == bull_regime, 1, 0)
                elif config.n_clusters == 3:
                    # Bull 레짐이면 2, Bear 레짐이면 0, 아니면 1
                    regime_pred = np.where(cluster == bull_regime, 2, 0)
                    regime_pred = np.where(cluster == bear_regime, 0, regime_pred)


                predictions.extend(regime_pred)
                true_returns.extend(y.numpy())
                dates.extend(date)
        else:
            for x, y, date, r in dataloader:
                x = x.to(config.device)
                if config.vae:
                    _, _, _, _, hidden, _ = model(x)
                else:
                    _, hidden = model(x, return_hidden=True)
                hidden = hidden.cpu().numpy()

                # 클러스터 할당
                cluster = kmeans.predict(hidden)

                if config.n_clusters == 2:
                    # Bull 레짐이면 1, 아니면 0
                    regime_pred = np.where(cluster == bull_regime, 1, 0)
                elif config.n_clusters == 3:
                    # Bull 레짐이면 2, Bear 레짐이면 0, 아니면 1
                    regime_pred = np.where(cluster == bull_regime, 2, 0)
                    regime_pred = np.where(cluster == bear_regime, 0, regime_pred)

                predictions.extend(regime_pred)
                true_returns.extend(r.numpy())
                dates.extend(date)

    return np.array(predictions), np.array(true_returns), dates