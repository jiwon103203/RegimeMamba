import argparse
import os
from datetime import datetime
from regime_mamba.config.config import RegimeMambaConfig
from regime_mamba.models.mamba_model import create_model_from_config
from regime_mamba.utils.utils import set_seed
from regime_mamba.evaluate.smoothing import find_optimal_filtering

def parse_args():
    """명령줄 인수 파싱"""
    parser = argparse.ArgumentParser(description='레짐 필터링 전략 최적화')
    
    parser.add_argument('--data_path', type=str, required=True, help='데이터 파일 경로')
    parser.add_argument('--model_path', type=str, required=True, help='모델 체크포인트 경로')
    parser.add_argument('--output_dir', type=str, default='./filtering_results', help='결과 저장 디렉토리')
    parser.add_argument('--transaction_cost', type=float, default=0.001, help='거래 비용 (0.001 = 0.1%)')
    parser.add_argument('--seed', type=int, default=42, help='랜덤 시드')
    
    # 모델 파라미터 인수
    parser.add_argument('--d_model', type=int, default=128, help='모델 차원')
    parser.add_argument('--d_state', type=int, default=128, help='상태 차원')
    parser.add_argument('--n_layers', type=int, default=4, help='레이어 수')
    parser.add_argument('--dropout', type=float, default=0.1, help='드롭아웃 비율')
    parser.add_argument('--seq_len', type=int, default=128, help='시퀀스 길이')
    parser.add_argument('--batch_size', type=int, default=64, help='배치 크기')
    parser.add_argument('--n_clusters', type=int, default=2, help='클러스터 수')
    
    parser.add_argument('--target_type', type=str)
    parser.add_argument('--target_horizon', type=int)
    
    return parser.parse_args()

def main():
    """메인 실행 함수"""
    # 명령줄 인수 파싱
    args = parse_args()
    
    # 시드 설정
    set_seed(args.seed)
    
    # 출력 디렉토리 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"filtering_test_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 설정 객체 생성
    config = RegimeMambaConfig()
    config.model_path = args.model_path
    config.transaction_cost = args.transaction_cost
    config.d_model = args.d_model
    config.d_state = args.d_state
    config.n_layers = args.n_layers
    config.dropout = args.dropout
    config.seq_len = args.seq_len
    config.batch_size = args.batch_size
    config.n_clusters = args.n_clusters
    config.target_type = args.target_type
    config.target_horizon = args.target_horizon
    
    # 파라미터 정보 저장
    with open(os.path.join(output_dir, 'parameters.txt'), 'w') as f:
        f.write(f"데이터 경로: {args.data_path}\n")
        f.write(f"모델 경로: {args.model_path}\n")
        f.write(f"거래 비용: {args.transaction_cost}\n")
        f.write(f"d_model: {args.d_model}\n")
        f.write(f"d_state: {args.d_state}\n")
        f.write(f"n_layers: {args.n_layers}\n")
        f.write(f"dropout: {args.dropout}\n")
        f.write(f"seq_len: {args.seq_len}\n")
        f.write(f"batch_size: {args.batch_size}\n")
        f.write(f"n_clusters: {args.n_clusters}\n")
        f.write(f"target_type: {args.target_type}\n")
        f.write(f"target_horizon: {args.target_horizon}\n")
    
    # 결과 그래프 저장 경로
    strategies_chart_path = os.path.join(output_dir, 'filtering_strategies_comparison.png')
    
    # 최적 필터링 찾기 실행
    print("다양한 레짐 필터링 전략 비교 분석 시작...")
    optimal_params, results = find_optimal_filtering(
        config, 
        args.data_path,
        save_path=strategies_chart_path
    )
    
    # 최적 파라미터 저장
    with open(os.path.join(output_dir, 'optimal_parameters.txt'), 'w') as f:
        f.write("최적 필터링 파라미터:\n")
        for param, value in optimal_params.items():
            f.write(f"{param}: {value}\n")
    
    print(f"분석 완료! 결과가 {output_dir}에 저장되었습니다.")

if __name__ == "__main__":
    main()
