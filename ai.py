import streamlit as st
import pandas as pd
import random
import time  # [추가] 응답 시간 측정을 위한 모듈

# 1. [함수 정의] 단순 평균(Moving Average) 필터
def apply_moving_average(raw_list):
    start_time = time.perf_counter() # 시간 측정 시작
    filtered_list = []
    for i in range(len(raw_list)):
        if i == 0 or i == len(raw_list) - 1:
            filtered_list.append(raw_list[i])
        else:
            # [변수/리스트 활용] 주변 3개 값의 평균 계산
            avg = (raw_list[i-1] + raw_list[i] + raw_list[i+1]) / 3.0
            filtered_list.append(1 if avg >= 0.5 else 0)
    end_time = time.perf_counter()
    return filtered_list, (end_time - start_time) * 1000 # 결과와 소요시간(ms) 반환

# 2. [함수 정의] 인공신경망 퍼셉트론(Perceptron) 필터
def apply_perceptron(raw_list):
    start_time = time.perf_counter()
    filtered_list = []
    # [변수 활용] 신경망 가중치와 편향 설정
    weights = [0.7, 1.2, 0.7] 
    bias = -1.0 
    
    for i in range(len(raw_list)):
        if i == 0 or i == len(raw_list) - 1:
            filtered_list.append(raw_list[i])
        else:
            # [조건문 활용] 활성화 함수(Step Function) 구현
            weighted_sum = (raw_list[i-1]*weights[0] + raw_list[i]*weights[1] + raw_list[i+1]*weights[2])
            output = 1 if (weighted_sum + bias) > 0 else 0
            filtered_list.append(output)
    end_time = time.perf_counter()
    return filtered_list, (end_time - start_time) * 1000

# 3. [함수 정의] 해밍 거리(Hamming Distance) 기반 에러 정정 필터
def apply_hamming_correction(raw_list):
    start_time = time.perf_counter()
    filtered_list = []
    # [리스트 활용] 이상적인 부울 패턴 정의
    valid_patterns = [[0, 0, 0], [1, 1, 1]] 
    
    # [반복문 활용] 전체 시퀀스 분석
    for i in range(len(raw_list)):
        if i == 0 or i == len(raw_list) - 1:
            filtered_list.append(raw_list[i])
        else:
            current_window = [raw_list[i-1], raw_list[i], raw_list[i+1]]
            distances = []
            
            # [조건문/반복문] 각 패턴과의 해밍 거리 계산
            for pattern in valid_patterns:
                dist = sum([1 for j in range(3) if current_window[j] != pattern[j]])
                distances.append(dist)
            
            # 거리가 가장 짧은 패턴의 비트로 복원
            best_match_idx = distances.index(min(distances))
            filtered_list.append(valid_patterns[best_match_idx][1])
            
    end_time = time.perf_counter()
    return filtered_list, (end_time - start_time) * 1000

def main():
    st.set_page_config(layout="wide", page_title="PLC Logic Optimizer")
    st.title("🏭 PLC 노이즈 필터링: 이산수학 vs 인공지능 성능 분석")
    
    st.markdown("""
    **탐구 목표:** PLC 센서 데이터의 노이즈를 제거할 때, **해밍 거리**와 **신경망(퍼셉트론)** 구조 중 어떤 것이 더 효율적인가?
    (연산 속도와 정확도를 동시에 비교하여 최적의 제어 로직을 탐색합니다.)
    """)
    st.divider()

    # [사용자 입력 처리] 사이드바 설정
    st.sidebar.header("🛠️ 시뮬레이션 환경 설정")
    noise_level = st.sidebar.slider("노이즈 확률 (%)", 0, 50, 20)
    test_size = st.sidebar.select_slider("테스트 데이터 크기 (연산 속도 측정용)", options=[1000, 5000, 10000, 50000])

    # [리스트/반복문 활용] 대량의 가상 센서 데이터 생성
    base_signal = ([0]*20 + [1]*20) * (test_size // 40)
    noisy_signal = []
    for val in base_signal:
        if random.randint(1, 100) <= noise_level:
            noisy_signal.append(1 - val)
        else:
            noisy_signal.append(val)

    # [함수 호출] 각 알고리즘 적용 및 시간 측정
    ma_result, ma_time = apply_moving_average(noisy_signal)
    nn_result, nn_time = apply_perceptron(noisy_signal)
    hd_result, hd_time = apply_hamming_correction(noisy_signal)

    # 정확도 계산 함수
    def get_acc(res, base):
        return (sum([1 for r, b in zip(res, base) if r == b]) / len(base)) * 100

    # --- 결과 시각화 섹션 ---
    col1, col2 = st.columns([2, 1])

    # --- 결과 시각화 섹션 ---
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📈 실시간 신호 복원 성능 (앞부분 200개 데이터)")
        st.caption("신호가 겹쳐서 보이지 않도록 필터별로 그래프를 분리했습니다.")

        # 1. 노이즈 상황 (정답 vs 노이즈)
        st.write("🔴 **1. 노이즈 발생 신호** (원본 데이터)")
        st.line_chart(pd.DataFrame({
            "정답(Ideal)": base_signal[:200],
            "노이즈(Noisy)": noisy_signal[:200]
        }), height=180)

        # 2. 해밍거리 결과 (정답 vs 해밍거리)
        st.write("🔵 **2. 해밍거리 정정 결과**")
        st.line_chart(pd.DataFrame({
            "정답(Ideal)": base_signal[:200],
            "해밍거리 정정": hd_result[:200]
        }), height=180)

        # 3. 퍼셉트론 결과 (정답 vs 퍼셉트론)
        st.write("🟢 **3. 퍼셉트론(NN) 필터링 결과**")
        st.line_chart(pd.DataFrame({
            "정답(Ideal)": base_signal[:200],
            "퍼셉트론 결과": nn_result[:200]
        }), height=180)

    with col2:
        # ... (이하 col2 코드는 기존과 동일하게 유지) ...
        st.subheader("⏱️ 연산 효율 및 정확도")
        # 데이터프레임으로 요약 결과 표시
        summary_data = {
            "알고리즘": ["단순 평균", "퍼셉트론(NN)", "해밍거리"],
            "정확도 (%)": [f"{get_acc(ma_result, base_signal):.2f}%", 
                         f"{get_acc(nn_result, base_signal):.2f}%", 
                         f"{get_acc(hd_result, base_signal):.2f}%"],
            "소요 시간 (ms)": [f"{ma_time:.2f}ms", f"{nn_time:.2f}ms", f"{hd_time:.2f}ms"]
        }
        st.table(pd.DataFrame(summary_data))
        
        # [조건문 활용] 가장 효율적인 알고리즘 추천
        fastest = min(ma_time, nn_time, hd_time)
        st.info(f"💡 현재 데이터 크기에서 가장 빠른 응답 속도는 **{fastest:.2f}ms** 입니다.")

    # [실제 문제 해결 결론]
    st.success(f"""
    ### 🔍 탐구 결론
    1. **정확도:** 해밍 거리와 퍼셉트론 모두 부울 대수 기반이므로 노이즈 제거 능력이 탁월합니다.
    2. **응답성:** {test_size}개의 데이터를 처리할 때, 복잡한 패턴 매칭(해밍 거리)보다 수치 연산(퍼셉트론/평균) 방식이 PLC 스캔 타임 단축에 유리할 수 있습니다.
    3. **안정성:** 해밍 거리는 비트 반전 오류에 대해 수학적 확신을 제공하므로 중요 세이프티 시스템에 더 적합합니다.
    """)

if __name__ == "__main__":
    main()
