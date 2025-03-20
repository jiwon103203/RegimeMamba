import argparse
from regime_mamba.evaluate.rolling_window import RollingWindowConfig, run_rolling_window_backtest
from regime_mamba.utils.utils import set_seed

def parse_args():
    """명령줄 인수 파싱"""
    parser = argparse.ArgumentParser(description='롤링 윈도우 백테스트 실행')
    
    parser.add_argument('--data_path', type=str, required=True, help='데이터 파일 경로')
    parser.add_argument('--model_path', type=str, required=True, help='사전 학습된 모델 경로')
    parser.add_argument('--results_dir', type=str, default='./rolling_window_results', help='결과 저장 디렉토리')
    parser.add_argument('--start_date', type=str, default='2010-01-01', help='백테스트 시작 날짜')
    parser.add_argument('--end_date', type=str, default='2023-12-31', help='백테스트 종료 날짜')
    parser.add_argument('--lookback_years', type=int, default=10, help='클러스터링을 위한 룩백 기간(년)')
    parser.add_argument('--forward_months', type=int, default=12, help='적용 기간(월)')
    parser.add_argument('--transaction_cost', type=float, default=0.001, help='거래 비용 (0.001 = 0.1%)')
    parser.add_argument('--seed', type=int, default=42, help='랜덤 시드')
    parser.add_argument('--preprocessed', type=bool, default=True)
    
    # 모델 파라미터 인수
    parser.add_argument('--d_model', type=int, default=128, help='모델 차원')
    parser.add_argument('--d_state', type=int, default=128, help='상태 차원')
    parser.add_argument('--n_layers', type=int, default=4, help='레이어 수')
    parser.add_argument('--dropout', type=float, default=0.1, help='드롭아웃 비율')
    
    return parser.parse_args()

def main():
    """메인 실행 함수"""
    # 명령줄 인수 파싱
    args = parse_args()
    
    # 시드 설정
    set_seed(args.seed)
    
    # 롤링 윈도우 설정 객체 생성
    config = RollingWindowConfig()
    config.data_path = args.data_path
    config.model_path = args.model_path
    config.results_dir = args.results_dir
    config.start_date = args.start_date
    config.end_date = args.end_date
    config.lookback_years = args.lookback_years
    config.forward_months = args.forward_months
    config.transaction_cost = args.transaction_cost
    config.preprocessed = args.preprocessed
    
    # 모델 파라미터 설정
    config.d_model = args.d_model
    config.d_state = args.d_state
    config.n_layers = args.n_layers
    config.dropout = args.dropout
    
    # 롤링 윈도우 백테스트 실행
    run_rolling_window_backtest(config, args.data_path)

if __name__ == "__main__":
    main()
