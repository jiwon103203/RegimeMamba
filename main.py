import os
import time
import json
import argparse
from regime_mamba.config.config import RegimeMambaConfig
from regime_mamba.data.dataset import create_dataloaders
from regime_mamba.models.mamba_model import create_model_from_config
from regime_mamba.train.train import train_regime_mamba
from regime_mamba.train.optimize import optimize_regime_mamba_bayesian
from regime_mamba.evaluate.clustering import extract_hidden_states, identify_bull_bear_regimes, predict_regimes
from regime_mamba.evaluate.strategy import evaluate_regime_strategy, analyze_transaction_cost_impact
from regime_mamba.utils.utils import set_seed

def parse_args():
    """명령줄 인수 파싱"""
    parser = argparse.ArgumentParser(description='RegimeMamba: Mamba 기반 시장 레짐 식별 시스템')
    
    parser.add_argument('--data_path', type=str, required=True, help='데이터 파일 경로')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='출력 디렉토리 경로')
    parser.add_argument('--optimize', action='store_true', help='하이퍼파라미터 최적화 수행 여부')
    parser.add_argument('--skip_training', action='store_true', help='훈련 건너뛰기 여부')
    parser.add_argument('--load_model', type=str, default=None, help='모델 체크포인트 로드 경로')
    parser.add_argument('--load_config', type=str, default=None, help='설정 파일 로드 경로')
    parser.add_argument('--transaction_cost', type=float, default=0.001, help='거래 비용 설정')
    parser.add_argument('--seed', type=int, default=42, help='랜덤 시드')
    parser.add_argument('--target_type, type=str, default="next_day")
    parser.add_argument('--target_horizon, type=int, default=1)

    args = parser.parse_args()
    return args

def main():
    """메인 실행 함수"""
    # 명령줄 인수 파싱
    args = parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 시드 설정
    set_seed(args.seed)

    # 시작 시간 기록
    start_time = time.time()
    
    # 기본 설정 초기화
    if args.load_config:
        config = RegimeMambaConfig.load(args.load_config)
    else:
        config = RegimeMambaConfig()
    
    config.data_path = args.data_path
    
    # 1. 하이퍼파라미터 최적화 (선택 사항)
    if args.optimize:
        print("1. 베이지안 최적화로 하이퍼파라미터 튜닝 중...")
        
        # 최적화 결과 저장 경로
        optimization_path = os.path.join(args.output_dir, 'bayesian_optimization_results.json')
        
        # 하이퍼파라미터 최적화 실행
        optimized_config = optimize_regime_mamba_bayesian(
            config.data_path, config, n_iterations=30, save_path=optimization_path
        )
        
        # 최적화된 설정 저장
        optimized_config_path = os.path.join(args.output_dir, 'optimized_config.json')
        optimized_config.save(optimized_config_path)
        
        # 최적화된 설정 사용
        config = optimized_config
        
        # 최적화 시간 기록
        optimization_time = time.time() - start_time
        print(f"최적화 완료 (소요 시간: {optimization_time/60:.1f}분)")
    
    # 2. 데이터로더 생성
    print("\n2. 데이터로더 생성 중...")
    train_loader, valid_loader, test_loader = create_dataloaders(config)
    print(f"훈련 데이터 배치 수: {len(train_loader)}")
    print(f"검증 데이터 배치 수: {len(valid_loader)}")
    print(f"테스트 데이터 배치 수: {len(test_loader)}")
    
    # 3. 모델 로드 또는 훈련
    if args.load_model:
        print(f"\n3. 기존 모델 로드 중: {args.load_model}")
        model = create_model_from_config(config)
        checkpoint = torch.load(args.load_model)
        model.load_state_dict(checkpoint['model_state_dict'])
    elif not args.skip_training:
        print("\n3. 모델 훈련 중...")
        model = create_model_from_config(config)
        
        # 모델 저장 경로
        model_save_path = os.path.join(args.output_dir, 'best_regime_mamba.pth')
        
        # 모델 훈련
        model = train_regime_mamba(model, train_loader, valid_loader, config, save_path=model_save_path)
    else:
        raise ValueError("모델을 로드하거나 훈련해야 합니다. --load_model 또는 --skip_training 중 하나를 선택하세요.")
    
    # 4. 레짐 클러스터링 및 평가
    print("\n4. 레짐 클러스터링 및 평가 중...")
    
    # 검증 데이터에서 hidden states 추출
    valid_hidden, valid_returns, valid_dates = extract_hidden_states(model, valid_loader, config.device)
    
    # Bull/Bear 레짐 식별
    kmeans, bull_regime = identify_bull_bear_regimes(valid_hidden, valid_returns, config)
    
    # 테스트 데이터에 대한 레짐 예측
    test_predictions, test_returns, test_dates = predict_regimes(
        model, test_loader, kmeans, bull_regime, config.device
    )
    
    # 5. 거래 비용을 고려한 레짐 기반 전략 성과 평가
    print(f"\n5. 거래 비용을 고려한 레짐 기반 전략 성과 평가 중 (거래 비용: {args.transaction_cost*100:.2f}%)...")
    
    # 전략 평가 그래프 저장 경로
    strategy_graph_path = os.path.join(args.output_dir, 'regime_strategy_performance.png')
    
    # 전략 평가 실행
    results_df, performance = evaluate_regime_strategy(
        test_predictions, test_returns, test_dates, 
        transaction_cost=args.transaction_cost,
        save_path=strategy_graph_path
    )
    
    # 결과 저장
    results_df.to_csv(os.path.join(args.output_dir, 'regime_strategy_detailed_results.csv'), index=False)
    
    with open(os.path.join(args.output_dir, 'regime_strategy_results_with_costs.json'), 'w') as f:
        json.dump(performance, f, indent=4)
    
    # 6. 거래 비용 영향 분석
    print("\n6. 다양한 거래 비용 수준에서 전략 성과 분석 중...")
    
    # 거래 비용 분석 그래프 저장 경로
    cost_analysis_graph_path = os.path.join(args.output_dir, 'transaction_cost_analysis.png')
    
    # 거래 비용 분석 실행
    cost_analysis_df = analyze_transaction_cost_impact(
        model, valid_loader, test_loader, config, kmeans, bull_regime,
        save_path=cost_analysis_graph_path
    )
    
    # 분석 결과 저장
    cost_analysis_df.to_csv(os.path.join(args.output_dir, 'transaction_cost_analysis.csv'), index=False)
    
    # 총 소요 시간 계산
    total_time = time.time() - start_time
    print(f"\n전체 과정 완료! 총 소요 시간: {total_time/60:.1f}분")
    
    return {
        'model': model,
        'config': config,
        'performance': performance
    }

if __name__ == "__main__":
    main()
