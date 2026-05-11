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
    test_size = st.sidebar.slider("테스트 데이터 크기", min_value=100, max_value=20000, value=1000, step=100)

    # 데이터 생성
    pattern = [0]*20 + [1]*20
    base_signal = (pattern * (test_size // 40 + 1))[:test_size]
    
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

    # --- 1. 상단: 시각화 및 결과 요약 ---
    col1, col2 = st.columns([2.5, 1])

    with col1:
        st.subheader("📈 실시간 신호 복원 상태 (로직 애널라이저 뷰)")
        fig = make_subplots(
            rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.05,
            subplot_titles=("1. 정답 신호", "2. 노이즈 신호", "3. 단순평균 필터", "4. 해밍거리 정정", "5. 퍼셉트론 정정")
        )

        fig.add_trace(go.Scatter(y=base_signal[:200], line=dict(color='#1f77b4', width=2, shape='hv'), name="정답"), row=1, col=1)
        fig.add_trace(go.Scatter(y=noisy_signal[:200], line=dict(color='#d62728', width=1.5, shape='hv'), name="노이즈"), row=2, col=1)
        fig.add_trace(go.Scatter(y=ma_result[:200], line=dict(color='#9467bd', width=2, shape='hv'), name="단순평균"), row=3, col=1)
        fig.add_trace(go.Scatter(y=hd_result[:200], line=dict(color='#2ca02c', width=2, shape='hv'), name="해밍거리"), row=4, col=1)
        fig.add_trace(go.Scatter(y=nn_result[:200], line=dict(color='#ff7f0e', width=2, shape='hv'), name="퍼셉트론"), row=5, col=1)

        fig.update_yaxes(tickvals=[0, 1], range=[-0.2, 1.2])
        fig.update_layout(height=800, showlegend=False, margin=dict(t=40, b=20, l=20, r=20))
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
        st.table(pd.DataFrame(summary_data))
        
        fastest = min(ma_time, nn_time, hd_time)
        st.info(f"💡 현재 {test_size}개 데이터 기준\n최단 응답 속도: **{fastest:.2f}ms**")

    # --- 2. 하단: 수학적 계산 원리 (LaTeX 수식 오류 수정됨) ---
    st.divider()
    st.header("🧮 알고리즘별 계산 방법 및 수학적 배경")
    st.markdown(r"수만 개의 데이터 중 연속된 3개의 윈도우 신호를 $x_{i-1}$ (과거), $x_i$ (현재), $x_{i+1}$ (미래)라고 할 때, 각 필터는 다음의 계산식을 거쳐 최종 출력 $y_i$를 결정합니다.")

    exp1, exp2, exp3 = st.columns(3)

    with exp1:
        st.info("📊 단순 평균 (Moving Average)")
        st.markdown("**계산 공식:**")
        st.markdown(r"$$y_i = \begin{cases} 1, & \text{if } \frac{x_{i-1} + x_i + x_{i+1}}{3} \ge 0.5 \\ 0, & \text{otherwise} \end{cases}$$")
        st.markdown("""
        **작동 원리:**
        3개 신호의 산술 평균을 구한 뒤, 반올림을 수행합니다. 
        즉, 3개 중 1이 두 번 이상 등장하면 1로 출력하는 **완벽한 다수결 논리**를 따릅니다.
        """)

    with exp2:
        st.success("🤖 퍼셉트론 (Perceptron)")
        st.markdown("**계산 공식:**")
        st.markdown(r"$$y_i = f\left(\sum_{j=1}^{3} w_j x_j + b\right)$$")
        st.markdown(f"""
        **현재 가중치 설정:**
        * $w$ (가중치) = `[0.7, 1.2, 0.7]`
        * $b$ (편향) = `-1.0`
        
        **작동 원리:**
        단순 다수결이 아니라, **현재 값($x_i$)에 가장 큰 가중치(1.2)**를 둡니다. 가중합이 0을 넘으면(활성화 함수) 1을 출력합니다.
        """)

    with exp3:
        st.warning("📐 해밍 거리 (Hamming Distance)")
        st.markdown("**계산 공식:**")
        st.markdown(r"$$D_H = \sum_{k=1}^{3} | u_k - v_k |$$")
        st.markdown("""
        **작동 원리:**
        입력된 3개의 신호를 이상적인 패턴 `[0,0,0]` 및 `[1,1,1]`과 비교합니다.
        서로 다른 비트의 개수(해밍 거리)를 세어, **거리가 더 짧은 (더 유사한) 패턴**의 값으로 에러를 정정합니다.
        """)

if __name__ == "__main__":
    main()
