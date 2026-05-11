import streamlit as st
import pandas as pd
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. 필터 알고리즘 정의 (매개변수 수신 가능하게 수정) ---
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

def apply_perceptron(raw_list, w1, w2, w3, b):
    start_time = time.perf_counter()
    filtered_list = []
    for i in range(len(raw_list)):
        if i == 0 or i == len(raw_list) - 1:
            filtered_list.append(raw_list[i])
        else:
            # 사용자가 설정한 가중치 적용
            w_sum = (raw_list[i-1]*w1 + raw_list[i]*w2 + raw_list[i+1]*w3) + b
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
    st.set_page_config(layout="wide", page_title="PLC Parameter Tuner")
    st.title("🏭 PLC 필터 파라미터 정밀 튜닝")
    st.markdown("가중치와 편향을 조절하여 현장 노이즈에 최적화된 AI 모델을 설계하세요.")
    st.divider()

    # --- 사이드바: 데이터 입력 및 튜닝 ---
    st.sidebar.header("📥 1. 데이터 입력")
    sample_data = "0000010000000101011111110111111111000000010000"
    custom_input = st.sidebar.text_area("센서 데이터 (0, 1)", value=sample_data, height=100)
    noisy_signal = [int(char) for char in custom_input if char in '01']
    
    st.sidebar.divider()
    
    st.sidebar.header("⚙️ 2. 퍼셉트론 파라미터 튜닝")
    st.sidebar.caption("각 입력값의 중요도와 판단 기준을 설정합니다.")
    
    w_past = st.sidebar.slider("이전 값 가중치 (W_past)", 0.0, 2.0, 0.7, 0.1)
    w_curr = st.sidebar.slider("현재 값 가중치 (W_curr)", 0.0, 2.0, 1.2, 0.1)
    w_next = st.sidebar.slider("다음 값 가중치 (W_next)", 0.0, 2.0, 0.7, 0.1)
    bias = st.sidebar.slider("판단 편향 (Bias)", -2.0, 0.0, -1.0, 0.1)
    
    st.sidebar.divider()
    calc_idx = st.sidebar.slider("분석 인덱스 선택", 1, len(noisy_signal)-2, 5)

    # --- 알고리즘 실행 ---
    ma_res, ma_time = apply_moving_average(noisy_signal)
    hd_res, hd_time = apply_hamming_correction(noisy_signal)
    # 튜닝된 파라미터 전달
    nn_res, nn_time = apply_perceptron(noisy_signal, w_past, w_curr, w_next, bias)

    # 신호 유지율 계산
    def get_stability(res, raw):
        match = sum([1 for r, o in zip(res, raw) if r == o])
        return (match / len(raw)) * 100

    # --- 메인 화면 레이아웃 ---
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📊 실시간 필터링 비교")
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05)
        titles = ["원본 입력", "단순 평균", "해밍 거리", f"퍼셉트론 (W:{w_past}/{w_curr}/{w_next}, B:{bias})"]
        signals = [noisy_signal, ma_res, hd_res, nn_res]
        colors = ['red', 'purple', 'green', 'orange']

        for i, (sig, clr, ttl) in enumerate(zip(signals, colors, titles)):
            fig.add_trace(go.Scatter(y=sig, line=dict(color=clr, width=2, shape='hv'), name=ttl), row=i+1, col=1)
            fig.add_vline(x=calc_idx, line_width=1, line_dash="dot", row=i+1, col=1)
        
        fig.update_layout(height=600, margin=dict(t=20, b=20), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📈 성능 리포트")
        report_df = pd.DataFrame({
            "알고리즘": ["단순 평균", "해밍 거리", "퍼셉트론(Tuned)"],
            "원본 유지율": [f"{get_stability(ma_res, noisy_signal):.1f}%", f"{get_stability(hd_res, noisy_signal):.1f}%", f"{get_stability(nn_res, noisy_signal):.1f}%"],
            "연산 속도(ms)": [f"{ma_time:.3f}", f"{hd_time:.3f}", f"{nn_time:.3f}"]
        })
        st.table(report_df)
        
        st.warning("**튜닝 가이드**")
        st.write("1. **노이즈가 그대로 통과된다면?** \n   - Bias를 더 낮은 음수(-1.5 등)로 조절하세요.")
        st.write("2. **반응 속도가 너무 느리다면?** \n   - W_curr(현재 값 가중치)를 높이세요.")

    # --- 하단: 상세 계산 프로세스 ---
    st.divider()
    st.header(f"🧮 인덱스 {calc_idx}의 실시간 수치 대입")
    v_p, v_c, v_n = noisy_signal[calc_idx-1], noisy_signal[calc_idx], noisy_signal[calc_idx+1]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### 1. 단순 평균")
        val = (v_p + v_c + v_n) / 3
        st.latex(rf"\frac{{{v_p} + {v_c} + {v_n}}}{{3}} = {val:.2f}")
        st.markdown(f"결과: **{'1' if val >= 0.5 else '0'}**")

    with c2:
        st.markdown("### 2. 해밍 거리")
        d0 = sum([1 for v, p in zip([v_p, v_c, v_n], [0,0,0]) if v != p])
        d1 = sum([1 for v, p in zip([v_p, v_c, v_n], [1,1,1]) if v != p])
        st.write(f"[0,0,0]과의 거리: {d0}")
        st.write(f"[1,1,1]과의 거리: {d1}")
        st.markdown(f"결과: **{'0' if d0 < d1 else '1'}**")

    with c3:
        st.markdown("### 3. 퍼셉트론 (튜닝값 적용)")
        # 실시간 슬라이더 값 반영
        w_sum = (v_p * w_past) + (v_c * w_curr) + (v_n * w_next) + bias
        st.latex(rf"({v_p} \cdot {w_past}) + ({v_c} \cdot {w_curr}) + ({v_n} \cdot {w_next}) + ({bias}) = {w_sum:.2f}")
        st.markdown(f"결과: **{'1' if w_sum > 0 else '0'}**")

if __name__ == "__main__":
    main()
