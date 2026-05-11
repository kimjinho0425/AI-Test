import streamlit as st
import pandas as pd
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. 필터 알고리즘 정의 ---
def apply_moving_average(raw_list):
    start_time = time.perf_counter()
    filtered_list = []
    for i in range(len(raw_list)):
        if i == 0 or i == len(raw_list) - 1:
            filtered_list.append(raw_list[i])
        else:
            avg = (raw_list[i-1] + raw_list[i] + raw_list[i+1]) / 3.0
            filtered_list.append(1 if avg >= 0.5 else 0)
    return filtered_list, (time.perf_counter() - start_time) * 1000

def apply_perceptron(raw_list):
    start_time = time.perf_counter()
    filtered_list = []
    weights = [0.7, 1.2, 0.7] 
    bias = -1.0 
    for i in range(len(raw_list)):
        if i == 0 or i == len(raw_list) - 1:
            filtered_list.append(raw_list[i])
        else:
            w_sum = (raw_list[i-1]*weights[0] + raw_list[i]*weights[1] + raw_list[i+1]*weights[2]) + bias
            filtered_list.append(1 if w_sum > 0 else 0)
    return filtered_list, (time.perf_counter() - start_time) * 1000

def apply_hamming_correction(raw_list):
    start_time = time.perf_counter()
    filtered_list = []
    for i in range(len(raw_list)):
        if i == 0 or i == len(raw_list) - 1:
            filtered_list.append(raw_list[i])
        else:
            window = [raw_list[i-1], raw_list[i], raw_list[i+1]]
            d0 = sum([1 for v in window if v != 0])
            d1 = sum([1 for v in window if v != 1])
            filtered_list.append(0 if d0 < d1 else 1)
    return filtered_list, (time.perf_counter() - start_time) * 1000

def main():
    st.set_page_config(layout="wide", page_title="PLC Real-Data Analyzer")
    st.title("🏭 PLC 실전 데이터 노이즈 필터링 검증")
    st.divider()

    # --- 사이드바: 데이터 입력 ---
    st.sidebar.header("📥 실제 로그 데이터 입력")
    sample_data = "0000100011110111" * 10 
    custom_input = st.sidebar.text_area("센서 데이터 (0, 1) 붙여넣기", value=sample_data, height=150)
    
    noisy_signal = [int(char) for char in custom_input if char in '01']
    test_size = len(noisy_signal)

    if test_size < 3:
        st.error("데이터가 너무 부족합니다.")
        st.stop()

    st.sidebar.subheader("🎯 분석 위치 설정")
    calc_idx = st.sidebar.slider("계산 과정을 볼 인덱스 선택", 1, test_size-2, value=min(5, test_size-2))

    # --- 알고리즘 실행 ---
    ma_res, ma_time = apply_moving_average(noisy_signal)
    nn_res, nn_time = apply_perceptron(noisy_signal)
    hd_res, hd_time = apply_hamming_correction(noisy_signal)

    # 신호 유지율(정확도 개념) 계산: 원본 대비 필터링 후 값이 얼마나 '정제'되었는지 지표
    def get_stability(res, raw):
        match = sum([1 for r, o in zip(res, raw) if r == o])
        return (match / len(raw)) * 100

    # --- 메인 화면 레이아웃 ---
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📊 필터링 시각화")
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05)
        titles = ["원본 입력", "단순 평균", "해밍 거리", "퍼셉트론"]
        signals = [noisy_signal, ma_res, hd_res, nn_res]
        colors = ['red', 'purple', 'green', 'orange']

        for i, (sig, clr, ttl) in enumerate(zip(signals, colors, titles)):
            fig.add_trace(go.Scatter(y=sig[:200], line=dict(color=clr, width=2, shape='hv'), name=ttl), row=i+1, col=1)
            fig.add_vline(x=calc_idx, line_width=1, line_dash="dot", row=i+1, col=1)
        
        fig.update_layout(height=600, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📈 성능 리포트")
        # 여기서 정확도는 '원본 데이터와의 일치율'을 기반으로 필터가 얼마나 공격적으로 수정했는지를 보여줍니다.
        report_df = pd.DataFrame({
            "알고리즘": ["단순 평균", "해밍 거리", "퍼셉트론"],
            "데이터 유지율(%)": [f"{get_stability(ma_res, noisy_signal):.1f}%", f"{get_stability(hd_res, noisy_signal):.1f}%", f"{get_stability(nn_res, noisy_signal):.1f}%"],
            "연산 속도(ms)": [f"{ma_time:.3f}", f"{hd_time:.3f}", f"{nn_time:.3f}"]
        })
        st.table(report_df)
        st.info(f"💡 **분석 결과**: 인덱스 {calc_idx}에서 입력값은 {noisy_signal[calc_idx]}입니다.")

    # --- 하단: 상세 계산 프로세스 (공식 및 수치 대입) ---
    st.divider()
    st.header(f"🧮 인덱스 {calc_idx}의 단계별 계산 과정")
    
    v_p, v_c, v_n = noisy_signal[calc_idx-1], noisy_signal[calc_idx], noisy_signal[calc_idx+1]
    st.markdown(f"**현재 관찰 윈도우:** `이전:{v_p}` , `현재:{v_c}` , `다음:{v_n}`")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("### 1. 단순 평균 (MA)")
        st.latex(r"y_i = \text{round}\left(\frac{x_{i-1} + x_i + x_{i+1}}{3}\right)")
        val = (v_p + v_c + v_n) / 3
        st.write(f"계산: ({v_p} + {v_c} + {v_n}) / 3 = **{val:.2f}**")
        st.write(f"결과: **{'1' if val >= 0.5 else '0'}**")

    with c2:
        st.markdown("### 2. 해밍 거리 (HD)")
        st.latex(r"D_H(x, pattern)")
        d0 = sum([1 for v, p in zip([v_p, v_c, v_n], [0,0,0]) if v != p])
        d1 = sum([1 for v, p in zip([v_p, v_c, v_n], [1,1,1]) if v != p])
        st.write(f"[0,0,0]과의 거리: **{d0}**")
        st.write(f"[1,1,1]과의 거리: **{d1}**")
        st.write(f"결과: **{'0' if d0 < d1 else '1'}**")

    with c3:
        st.markdown("### 3. 퍼셉트론 (NN)")
        st.latex(r"\sigma(\sum W_i X_i + b)")
        w_sum = (v_p * 0.7) + (v_c * 1.2) + (v_n * 0.7) - 1.0
        st.write(f"가중합: ({v_p}×0.7) + ({v_c}×1.2) + ({v_n}×0.7) - 1.0 = **{w_sum:.1f}**")
        st.write(f"결과: **{'1' if w_sum > 0 else '0'}**")

if __name__ == "__main__":
    main()
