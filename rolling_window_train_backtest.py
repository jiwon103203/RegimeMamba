import os
import sys

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import os
from datetime import datetime
from regime_mamba.evaluate.rolling_window_w_train import RollingWindowTrainConfig, run_rolling_window_train
from regime_mamba.utils.utils import set_seed

def parse_args():
    """명령줄 인수 파싱"""
    parser = argparse.ArgumentParser(description='롤링 윈도우 재학습 백테스트 실행')
    
    parser.add_argument('--data_path', type=str, required=True, help='데이터 파일 경로')
    parser.add_argument('--results_dir', type=str, default='./rolling_window_train_results', help='결과 저장 디렉토리')
    parser.add_argument('--start_date', type=str, default='1990-01-01', help='시작 날짜')
    parser.add_argument('--end_date', type=str, default='2023-12-31', help='종료 날짜')
    
    # 기간 관련 설정
    parser.add_argument('--total_window_years', type=int, default=40, help='총 사용할 데이터 기간(년)')
    parser.add_argument('--train_years', type=int, default=20, help='학습에 사용할 기간(년)')
    parser.add_argument('--valid_years', type=int, default=10, help='검증에 사용할 기간(년)')
    parser.add_argument('--clustering_years', type=int, default=10, help='클러스터링에 사용할 기간(년)')
    parser.add_argument('--forward_months', type=int, default=60, help='다음 윈도우까지의 간격(개월)')
    
    # 모델 파라미터
    parser.add_argument('--d_model', type=int, default=128, help='모델 차원')
    parser.add_argument('--d_state', type=int, default=128, help='상태 차원')
    parser.add_argument('--n_layers', type=int, default=4, help='레이어 수')
    parser.add_argument('--dropout', type=float, default=0.1, help='드롭아웃 비율')
    parser.add_argument('--seq_len', type=int, default=128, help='시퀀스 길이')
    parser.add_argument('--batch_size', type=int, default=64, help='배치 크기')
    parser.add_argument('--learning_rate', type=float, default=1e-6, help='학습률')
    
    # 학습 관련 설정
    parser.add_argument('--max_epochs', type=int, default=100, help='최대 학습 에폭')
    parser.add_argument('--patience', type=int, default=10, help='조기 종료 인내심')
    parser.add_argument('--no_onecycle', action='store_false', dest='use_onecycle', help='OneCycleLR 사용 안함')
    
    # 필터링 관련 설정
    parser.add_argument('--no_filtering', action='store_false', dest='apply_filtering', help='레짐 필터링 적용 안함')
    parser.add_argument('--filter_method', type=str, default='minimum_holding', 
                        choices=['minimum_holding', 'smoothing'], help='필터링 방법')
    parser.add_argument('--min_holding_days', type=int, default=20, help='최소 보유 기간(일)')
    
    # 기타 설정
    parser.add_argument('--transaction_cost', type=float, default=0.001, help='거래 비용 (0.001 = 0.1%)')
    parser.add_argument('--seed', type=int, default=42, help='랜덤 시드')
    
    parser.set_defaults(use_onecycle=True, apply_filtering=True)
    
    return parser.parse_args()

def main():
    """메인 실행 함수"""
    # 명령줄 인수 파싱
    args = parse_args()
    
    # 시드 설정
    set_seed(args.seed)
    
    # 출력 디렉토리 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.results_dir, f"rolling_train_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 설정 객체 생성
    config = RollingWindowTrainConfig()
    config.data_path = args.data_path
    config.results_dir = output_dir
    config.start_date = args.start_date
    config.end_date = args.end_date
    
    # 기간 관련 설정
    config.total_window_years = args.total_window_years
    config.train_years = args.train_years
    config.valid_years = args.valid_years
    config.clustering_years = args.clustering_years
    config.forward_months = args.forward_months
    
    # 모델 파라미터
    config.d_model = args.d_model
    config.d_state = args.d_state
    config.n_layers = args.n_layers
    config.dropout = args.dropout
    config.seq_len = args.seq_len
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    
    # 학습 관련 설정
    config.max_epochs = args.max_epochs
    config.patience = args.patience
    config.use_onecycle = args.use_onecycle
    
    # 필터링 관련 설정
    config.apply_filtering = args.apply_filtering
    config.filter_method = args.filter_method
    config.min_holding_days = args.min_holding_days
    
    # 기타 설정
    config.transaction_cost = args.transaction_cost
    
    # 설정 정보 저장
    with open(os.path.join(output_dir, 'config.txt'), 'w') as f:
        f.write("=== 롤링 윈도우 재학습 설정 ===\n")
        f.write(f"데이터 경로: {config.data_path}\n")
        f.write(f"시작 날짜: {config.start_date}\n")
        f.write(f"종료 날짜: {config.end_date}\n")
        f.write(f"총 데이터 기간: {config.total_window_years}년\n")
        f.write(f"학습 기간: {config.train_years}년\n")
        f.write(f"검증 기간: {config.valid_years}년\n")
        f.write(f"클러스터링 기간: {config.clustering_years}년\n")
        f.write(f"다음 윈도우 간격: {config.forward_months}개월\n")
        f.write(f"모델 차원: {config.d_model}\n")
        f.write(f"상태 차원: {config.d_state}\n")
        f.write(f"레이어 수: {config.n_layers}\n")
        f.write(f"드롭아웃 비율: {config.dropout}\n")
        f.write(f"시퀀스 길이: {config.seq_len}\n")
        f.write(f"배치 크기: {config.batch_size}\n")
        f.write(f"학습률: {config.learning_rate}\n")
        f.write(f"최대 에폭: {config.max_epochs}\n")
        f.write(f"조기 종료 인내심: {config.patience}\n")
        f.write(f"OneCycleLR 사용: {config.use_onecycle}\n")
        f.write(f"레짐 필터링 적용: {config.apply_filtering}\n")
        if config.apply_filtering:
            f.write(f"필터링 방법: {config.filter_method}\n")
            if config.filter_method == 'minimum_holding':
                f.write(f"최소 보유 기간: {config.min_holding_days}일\n")
        f.write(f"거래 비용: {config.transaction_cost}\n")
    
    # 롤링 윈도우 재학습 실행
    print("=== 롤링 윈도우 재학습 백테스트 시작 ===")
    combined_results, all_performances, model_histories = run_rolling_window_train(config)
    
    if combined_results is not None:
        print(f"롤링 윈도우 재학습 완료! 결과가 {output_dir}에 저장되었습니다.")
    else:
        print("롤링 윈도우 재학습 실패!")

if __name__ == "__main__":
    main()