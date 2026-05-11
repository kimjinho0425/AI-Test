import streamlit as st
import pandas as pd
import random
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. 단순 평균(Moving Average) 필터
def apply_moving_average(raw_list):
    start_time = time.perf_counter()
    filtered_list = []
    for i in range(len(raw_list)):
        if i == 0 or i == len(raw_list) - 1:
            filtered_list.append(raw_list[i])
        else:
            avg = (raw_list[i-1] + raw_list[i] + raw_list[i+1]) / 3.0
            filtered_list.append(1 if avg >= 0.5 else 0)
    end_time = time.perf_counter()
    return filtered_list, (end_time - start_time) * 1000

# 2. 인공신경망 퍼셉트론(Perceptron) 필터
def apply_perceptron(raw_list):
    start_time = time.perf_counter()
    filtered_list = []
    weights = [0.7, 1.2, 0.7] 
    bias = -1.0 
    
    for i in range(len(raw_list)):
        if i == 0 or i == len(raw_list) - 1:
            filtered_list.append(raw_list[i])
        else:
            weighted_sum = (raw_list[i-1]*weights[0] + raw_list[i]*weights[1] + raw_list[i+1]*weights[2])
            output = 1 if (weighted_sum + bias) > 0 else 0
            filtered_list.append(output)
    end_time = time.perf_counter()
    return filtered_list, (end_time - start_time) * 1000

# 3. 해밍 거리(Hamming Distance) 기반 에러 정정 필터
def apply_hamming_correction(raw_list):
    start_time = time.perf_counter()
    filtered_list = []
    valid_patterns = [[0, 0, 0], [1, 1, 1]] 
    
    for i in range(len(raw_list)):
        if i == 0 or i == len(raw_list) - 1:
            filtered_list.append(raw_list[i])
        else:
            current_window = [raw_list[i-1], raw_list[i], raw_list[i+1]]
            distances = []
            
            for pattern in valid_patterns:
                dist = sum([1 for j in range(3) if current_window[j] != pattern[j]])
                distances.append(dist)
            
            best_match_idx = distances.index(min(distances))
            filtered_list.append(valid_patterns[best_match_idx][1])
            
    end_time = time.perf_counter()
    return filtered_list, (end_time - start_time) * 1000

def main():
    st.set_page_config(layout="wide", page_title="PLC Logic Optimizer")
    st.title("🏭 PLC 노이즈 필터링 성능 분석")
    st.divider()

    # 사이드바 설정
    st.sidebar.header("🛠️ 시뮬레이션 환경 설정")
    noise_level = st.sidebar.slider("노이즈 확률 (%)", 0, 50, 20)
    test_size = st.sidebar.select_slider("테스트 데이터 크기", options=[1000, 5000, 10000, 50000])

    # 데이터 생성
    base_signal = ([0]*20 + [1]*20) * (test_size // 40)
    noisy_signal = []
    for val in base_signal:
        if random.randint(1, 100) <= noise_level:
            noisy_signal.append(1 - val)
        else:
            noisy_signal.append(val)

    # 필터 적용
    ma_result, ma_time = apply_moving_average(noisy_signal)
    nn_result, nn_time = apply_perceptron(noisy_signal)
    hd_result, hd_time = apply_hamming_correction(noisy_signal)

    def get_acc(res, base):
        return (sum([1 for r, b in zip(res, base) if r == b]) / len(base)) * 100

    # --- 시각화 섹션 (가시성 대폭 개선) ---
    col1, col2 = st.columns([2.5, 1])

    with col1:
        st.subheader("📈 실시간 신호 복원 상태 (로직 애널라이저 뷰)")
        
        # Plotly를 이용한 다중 계단형(Step) 차트 생성
        fig = make_subplots(
            rows=4, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.08,
            subplot_titles=("1. 정답 신호 (파란색)", "2. 노이즈 발생 신호 (빨간색)", "3. 해밍거리 정정 (초록색)", "4. 퍼셉트론 정정 (주황색)")
        )

        # shape='hv' 옵션이 디지털 신호를 직각(계단형)으로 그려줍니다.
        fig.add_trace(go.Scatter(y=base_signal[:200], line=dict(color='#1f77b4', width=2, shape='hv'), name="정답"), row=1, col=1)
        fig.add_trace(go.Scatter(y=noisy_signal[:200], line=dict(color='#d62728', width=1.5, shape='hv'), name="노이즈"), row=2, col=1)
        fig.add_trace(go.Scatter(y=hd_result[:200], line=dict(color='#2ca02c', width=2, shape='hv'), name="해밍거리"), row=3, col=1)
        fig.add_trace(go.Scatter(y=nn_result[:200], line=dict(color='#ff7f0e', width=2, shape='hv'), name="퍼셉트론"), row=4, col=1)

        # Y축을 0과 1로 고정하여 깔끔하게 표시
        fig.update_yaxes(tickvals=[0, 1], range=[-0.2, 1.2])
        fig.update_layout(height=650, showlegend=False, margin=dict(t=40, b=20, l=20, r=20))
        
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("⏱️ 연산 효율 및 정확도")
        summary_data = {
            "알고리즘": ["단순 평균", "퍼셉트론(NN)", "해밍거리"],
            "정확도 (%)": [f"{get_acc(ma_result, base_signal):.2f}%", 
                         f"{get_acc(nn_result, base_signal):.2f}%", 
                         f"{get_acc(hd_result, base_signal):.2f}%"],
            "시간 (ms)": [f"{ma_time:.2f}", f"{nn_time:.2f}", f"{hd_time:.2f}"]
        }
        st.dataframe(pd.DataFrame(summary_data), hide_index=True)
        
        fastest = min(ma_time, nn_time, hd_time)
        st.info(f"💡 가장 빠른 응답 속도: **{fastest:.2f}ms**")

if __name__ == "__main__":
    main()
