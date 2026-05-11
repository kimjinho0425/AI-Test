import streamlit as st
import pandas as pd
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. 필터 알고리즘 정의 (동일) ---
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

def apply_hamming_correction(raw_list):
    start_time = time.perf_counter()
    filtered_list = []
    valid_patterns = [[0, 0, 0], [1, 1, 1]] 
    for i in range(len(raw_list)):
        if i == 0 or i == len(raw_list) - 1:
            filtered_list.append(raw_list[i])
        else:
            current_window = [raw_list[i-1], raw_list[i], raw_list[i+1]]
            distances = [sum([1 for j in range(3) if current_window[j] != p[j]]) for p in valid_patterns]
            best_match_idx = distances.index(min(distances))
            filtered_list.append(valid_patterns[best_match_idx][1])
    end_time = time.perf_counter()
    return filtered_list, (end_time - start_time) * 1000

def main():
    st.set_page_config(layout="wide", page_title="PLC Real-Data Analyzer")
    st.title("🏭 실전 PLC 센서 노이즈 분석 및 필터링")
    st.markdown("현장에서 수집된 실제 바이너리(0, 1) 데이터를 입력하여 필터 성능을 검증합니다.")
    st.divider()

    # --- 사이드바: 데이터 입력 및 컨트롤 ---
    st.sidebar.header("📥 데이터 입력")
    # 기본 예시 (실제 현장에서 흔히 발생하는 튀는 노이즈 패턴)
    sample_data = "0000010000000000111110111111110111111111" * 5
    custom_input = st.sidebar.text_area("센서 로그 붙여넣기 (0, 1)", value=sample_data, height=200)
    
    # 데이터 전처리 (0과 1만 추출)
    noisy_signal = [int(char) for char in custom_input if char in '01']
    test_size = len(noisy_signal)

    if test_size < 3:
        st.error("데이터가 너무 적습니다. 최소 3개 이상의 0 또는 1을 입력해주세요.")
        st.stop()

    st.sidebar.success(f"분석 중인 데이터: {test_size} bits")
    
    st.sidebar.divider()
    st.sidebar.subheader("🔍 정밀 분석 위치")
    calc_idx = st.sidebar.slider("수치 계산을 확인할 인덱스", 1, test_size-2, value=min(15, test_size-2))

    # --- 알고리즘 실행 ---
    ma_result, ma_time = apply_moving_average(noisy_signal)
    nn_result, nn_time = apply_perceptron(noisy_signal)
    hd_result, hd_time = apply_hamming_correction(noisy_signal)

    # --- 메인 화면 레이아웃 ---
    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("📊 필터링 결과 비교")
        fig = make_subplots(
            rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.08,
            subplot_titles=("원본 노이즈 신호", "단순평균 필터", "해밍거리 정정", "퍼셉트론(AI) 정정")
        )

        # 그래프 표시 범위 (너무 길면 느려지므로 앞부분 300개 위주로 표시)
        limit = min(300, test_size)
        
        colors = ['#d62728', '#9467bd', '#2ca02c', '#ff7f0e']
        signals = [noisy_signal, ma_result, hd_result, nn_result]
        names = ["Input", "Moving Avg", "Hamming", "Perceptron"]

        for i, (sig, color, name) in enumerate(zip(signals, colors, names)):
            fig.add_trace(go.Scatter(y=sig[:limit], line=dict(color=color, width=2, shape='hv'), name=name), row=i+1, col=1)
            # 선택된 위치 세로선 표시
            if calc_idx < limit:
                fig.add_vline(x=calc_idx, line_width=2, line_dash="dash", line_color="black", row=i+1, col=1)

        fig.update_yaxes(tickvals=[0, 1], range=[-0.2, 1.2])
        fig.update_layout(height=700, showlegend=False, margin=dict(t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("⚡ 처리 속도 리포트")
        perf_data = {
            "필터 종류": ["단순 평균", "해밍 거리", "퍼셉트론"],
            "소요 시간(ms)": [f"{ma_time:.3f}", f"{hd_time:.3f}", f"{nn_time:.3f}"]
        }
        st.table(pd.DataFrame(perf_data))
        
        st.info(f"**데이터 요약**\n- 총 데이터: {test_size} bits\n- 처리 지연: {min(ma_time, nn_time, hd_time):.3f}ms")
        st.caption("※ 실제 PLC 환경에서는 이 연산 속도가 실시간 제어의 핵심 지표가 됩니다.")

    # --- 하단: 실제 수치 대입 계산기 ---
    st.divider()
    st.header(f"🧮 인덱스 {calc_idx}번 데이터의 실제 수치 계산")
    
    ex_past, ex_curr, ex_next = noisy_signal[calc_idx-1], noisy_signal[calc_idx], noisy_signal[calc_idx+1]
    st.info(f"선택된 윈도우 데이터: **[ {ex_past}, {ex_curr}, {ex_next} ]**")

    math_col1, math_col2, math_col3 = st.columns(3)

    with math_col1:
        st.markdown("**1. 단순 평균 계산**")
        avg = (ex_past + ex_curr + ex_next) / 3
        st.latex(rf"\frac{{{ex_past} + {ex_curr} + {ex_next}}}{{3}} = {avg:.2f}")
        st.markdown(f"결과: **{'1' if avg >= 0.5 else '0'}**")

    with math_col2:
        st.markdown("**2. 퍼셉트론 가중치 계산**")
        w_sum = (ex_past*0.7) + (ex_curr*1.2) + (ex_next*0.7) - 1.0
        st.latex(rf"({ex_past} \cdot 0.7) + ({ex_curr} \cdot 1.2) + ({ex_next} \cdot 0.7) - 1.0 = {w_sum:.1f}")
        st.markdown(f"결과: **{'1' if w_sum > 0 else '0'}**")

    with math_col3:
        st.markdown("**3. 해밍 거리 패턴 매칭**")
        d0 = sum([1 for v, p in zip([ex_past, ex_curr, ex_next], [0,0,0]) if v != p])
        d1 = sum([1 for v, p in zip([ex_past, ex_curr, ex_next], [1,1,1]) if v != p])
        st.markdown(f"- [0,0,0]과의 거리: {d0}")
        st.markdown(f"- [1,1,1]과의 거리: {d1}")
        st.markdown(f"결과: **{'0' if d0 < d1 else '1'}**")

if __name__ == "__main__":
    main()
