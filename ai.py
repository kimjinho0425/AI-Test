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
    
    # 입력 모드 선택기 추가
    input_mode = st.sidebar.radio(
        "데이터 입력 방식 선택",
        ("🎲 랜덤 확률 시뮬레이션", "📝 실제 센서 데이터 직접 기입")
    )

    if input_mode == "🎲 랜덤 확률 시뮬레이션":
        noise_level = st.sidebar.slider("노이즈 확률 (%)", 0, 50, 20)
        test_size = st.sidebar.slider("테스트 데이터 크기", min_value=100, max_value=20000, value=1000, step=100)
        
        # 이상적인 정답 신호 생성 (주기 40)
        pattern = [0]*20 + [1]*20
        base_signal = (pattern * (test_size // 40 + 1))[:test_size]
        
        noisy_signal = []
        for val in base_signal:
            if random.randint(1, 100) <= noise_level:
                noisy_signal.append(1 - val)
            else:
                noisy_signal.append(val)
    else:
        st.sidebar.markdown("현장에서 추출한 `0`과 `1`의 연속된 데이터를 아래에 붙여넣기 하세요. (공백이나 줄바꿈은 무시됩니다)")
        
        # 기본 예시 데이터 제공
        default_custom_data = "0000000100000000000011111101111111111111" * 10 
        custom_input = st.sidebar.text_area("실제 노이즈 데이터 기입", value=default_custom_data, height=150)
        
        # 입력된 텍스트에서 0과 1만 추출하여 리스트로 변환
        noisy_signal = [int(char) for char in custom_input if char in '01']
        
        if len(noisy_signal) < 10:
            st.error("⚠️ 데이터를 분석하려면 최소 10개 이상의 0과 1이 필요합니다!")
            st.stop()
            
        test_size = len(noisy_signal)
        
        # 정확도 비교를 위해 입력 길이와 똑같은 '정상 신호'를 가상으로 깔아줌
        pattern = [0]*20 + [1]*20
        base_signal = (pattern * (test_size // 40 + 1))[:test_size]
        
        st.sidebar.success(f"✅ 총 {test_size}개의 실제 데이터가 정상적으로 인식되었습니다!")

    st.sidebar.divider()
    st.sidebar.subheader("🧮 상세 계산 데이터 선택")
    st.sidebar.markdown("아래 슬라이더를 움직여 특정 시점의 데이터 계산 과정을 확인하세요.")
    
    # 데이터 크기에 맞춰 슬라이더 최대값 자동 조절 방어코드
    max_slider_val = max(1, test_size - 2)
    calc_idx = st.sidebar.slider("계산해볼 데이터 인덱스", min_value=1, max_value=max_slider_val, value=min(15, max_slider_val))

    # 선택된 인덱스의 과거, 현재, 미래 데이터 추출
    ex_past = noisy_signal[calc_idx - 1]
    ex_curr = noisy_signal[calc_idx]
    ex_next = noisy_signal[calc_idx + 1]
    ex_base = base_signal[calc_idx]

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
            subplot_titles=("1. 정답 신호 기준 (참고용)", "2. 입력된 노이즈 신호", "3. 단순평균 필터", "4. 해밍거리 정정", "5. 퍼셉트론 정정")
        )

        view_limit = min(200, test_size) # 최대 200개까지만 그래프에 표시
        
        fig.add_trace(go.Scatter(y=base_signal[:view_limit], line=dict(color='#1f77b4', width=2, shape='hv'), name="정답"), row=1, col=1)
        fig.add_trace(go.Scatter(y=noisy_signal[:view_limit], line=dict(color='#d62728', width=1.5, shape='hv'), name="노이즈"), row=2, col=1)
        fig.add_trace(go.Scatter(y=ma_result[:view_limit], line=dict(color='#9467bd', width=2, shape='hv'), name="단순평균"), row=3, col=1)
        fig.add_trace(go.Scatter(y=hd_result[:view_limit], line=dict(color='#2ca02c', width=2, shape='hv'), name="해밍거리"), row=4, col=1)
        fig.add_trace(go.Scatter(y=nn_result[:view_limit], line=dict(color='#ff7f0e', width=2, shape='hv'), name="퍼셉트론"), row=5, col=1)

        # 선택된 인덱스 위치에 세로선 표시
        if calc_idx < view_limit:
            for row in range(1, 6):
                fig.add_vline(x=calc_idx, line_width=2, line_dash="dash", line_color="black", row=row, col=1)

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
        if input_mode == "📝 실제 센서 데이터 직접 기입":
            st.caption("※ 실제 데이터 입력 시, 정확도(%)는 가상의 이상적인 신호를 기준으로 계산된 참고 수치입니다.")

    # --- 2. 하단: 수학적 계산 원리 및 실시간 동적 계산 ---
    st.divider()
    st.header("🧮 알고리즘별 계산 방법 및 실시간 데이터 검증")
    st.markdown(f"사이드바에서 선택한 **인덱스 {calc_idx}번**의 윈도우 데이터를 각 수식에 대입하여 실시간으로 계산합니다.")
    
    st.code(f"현재 센서 입력 윈도우: [ {ex_past}, {ex_curr}, {ex_next} ]", language="text")

    exp1, exp2, exp3 = st.columns(3)

    with exp1:
        st.info("📊 단순 평균 (Moving Average)")
        st.markdown(r"$$y_i = \begin{cases} 1, & \text{if } \frac{x_{i-1} + x_i + x_{i+1}}{3} \ge 0.5 \\ 0, & \text{otherwise} \end{cases}$$")
        
        avg_val = (ex_past + ex_curr + ex_next) / 3.0
        ma_final = 1 if avg_val >= 0.5 else 0
        
        st.markdown("**🔍 실시간 계산 결과:**")
        st.markdown(f"1. 세 값의 합: **{ex_past} + {ex_curr} + {ex_next}** = **{ex_past+ex_curr+ex_next}**")
        st.markdown(f"2. 평균 계산: **{ex_past+ex_curr+ex_next} / 3** = **{avg_val:.2f}**")
        st.markdown(f"3. 판정: **{avg_val:.2f}**는 0.5 이상인가? {'**예**' if avg_val >= 0.5 else '**아니오**'}")
        st.markdown(f"👉 **최종 출력값: {ma_final}**")

    with exp2:
        st.success("🤖 퍼셉트론 (Perceptron)")
        st.markdown(r"$$y_i = f\left(\sum_{j=1}^{3} w_j x_j + b\right)$$")
        
        w_sum = (ex_past*0.7) + (ex_curr*1.2) + (ex_next*0.7) - 1.0
        nn_final = 1 if w_sum > 0 else 0
        
        st.markdown("**🔍 실시간 계산 결과:**")
        st.markdown(f"1. 가중치 곱셈: ({ex_past}×0.7) + ({ex_curr}×1.2) + ({ex_next}×0.7)")
        st.markdown(f"2. 편향(Bias) 더하기: **{(ex_past*0.7) + (ex_curr*1.2) + (ex_next*0.7):.2f} - 1.0**")
        st.markdown(f"3. 가중합 결과: **{w_sum:.2f}**")
        st.markdown(f"4. 활성화 함수: **{w_sum:.2f}** > 0 인가? {'**예**' if w_sum > 0 else '**아니오**'}")
        st.markdown(f"👉 **최종 출력값: {nn_final}**")

    with exp3:
        st.warning("📐 해밍 거리 (Hamming Distance)")
        st.markdown(r"$$D_H = \sum_{k=1}^{3} | u_k - v_k |$$")
        
        dist_0 = sum([1 for j, v in enumerate([ex_past, ex_curr, ex_next]) if v != [0,0,0][j]])
        dist_1 = sum([1 for j, v in enumerate([ex_past, ex_curr, ex_next]) if v != [1,1,1][j]])
        hd_final = 0 if dist_0 < dist_1 else 1
        
        st.markdown("**🔍 실시간 계산 결과:**")
        st.markdown(f"현재 윈도우: `[{ex_past}, {ex_curr}, {ex_next}]`")
        st.markdown(f"1. `[0,0,0]`과의 거리: 다른 비트 **{dist_0}개**")
        st.markdown(f"2. `[1,1,1]`과의 거리: 다른 비트 **{dist_1}개**")
        st.markdown(f"3. 비교: 어느 패턴과 더 가까운가? **{hd_final}**")
        st.markdown(f"👉 **최종 출력값: {hd_final}**")

if __name__ == "__main__":
    main()
