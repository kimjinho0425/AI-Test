import streamlit as st
import pandas as pd
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. 필터 알고리즘 정의 (기존 필터들 유지) ---
def apply_moving_average(raw_list):
    start_time = time.perf_counter()
    res = [raw_list[i] if i == 0 or i == len(raw_list)-1 else (1 if (sum(raw_list[i-1:i+2])/3.0) >= 0.5 else 0) for i in range(len(raw_list))]
    return res, (time.perf_counter() - start_time) * 1000

def apply_hamming_correction(raw_list):
    start_time = time.perf_counter()
    res = []
    for i in range(len(raw_list)):
        if i == 0 or i == len(raw_list) - 1:
            res.append(raw_list[i])
        else:
            window = [raw_list[i-1], raw_list[i], raw_list[i+1]]
            d0 = sum([1 for v in window if v != 0]) # 000과의 거리
            d1 = sum([1 for v in window if v != 1]) # 111과의 거리
            res.append(0 if d0 < d1 else 1)
    return res, (time.perf_counter() - start_time) * 1000

def apply_perceptron(raw_list, w1, w2, w3, b):
    start_time = time.perf_counter()
    res = []
    for i in range(len(raw_list)):
        if i == 0 or i == len(raw_list) - 1:
            res.append(raw_list[i])
        else:
            w_sum = (raw_list[i-1]*w1 + raw_list[i]*w2 + raw_list[i+1]*w3) + b
            res.append(1 if w_sum > 0 else 0)
    return res, (time.perf_counter() - start_time) * 1000

# --- 2. 분석 지표 계산 함수 ---
def get_stability(res, raw):
    match = sum([1 for r, o in zip(res, raw) if r == o])
    return (match / len(raw)) * 100

def get_roughness(signal):
    return sum([abs(signal[i] - signal[i-1]) for i in range(1, len(signal))])

def main():
    st.set_page_config(layout="wide", page_title="PLC Filter Expert")
    st.title("🏭 PLC 필터 정밀 진단 시스템 (통합 버전)")
    st.markdown("기존 필터링 성능 지표와 새로운 신호 거칠기 분석을 결합한 통합 대시보드입니다.")
    st.divider()

    # --- 사이드바 ---
    st.sidebar.header("📥 데이터 및 파라미터")
    sample_data = "0000010000000101011111110111111111000000010000"
    custom_input = st.sidebar.text_area("센서 데이터 (0, 1)", value=sample_data, height=100)
    raw = [int(c) for c in custom_input if c in '01']
    
    st.sidebar.divider()
    w_p = st.sidebar.slider("W_past", 0.0, 2.0, 0.7, 0.1)
    w_c = st.sidebar.slider("W_curr", 0.0, 2.0, 1.2, 0.1)
    w_n = st.sidebar.slider("W_next", 0.0, 2.0, 0.7, 0.1)
    bias = st.sidebar.slider("Bias", -2.0, 0.0, -1.0, 0.1)

    if st.sidebar.button("🚀 최적 Bias 자동 탐색"):
        best_r = float('inf')
        best_b = bias
        for test_b in [x * -0.1 for x in range(0, 21)]:
            res, _ = apply_perceptron(raw, w_p, w_c, w_n, test_b)
            r = get_roughness(res)
            if r < best_r:
                best_r = r
                best_b = test_b
        st.sidebar.success(f"추천 Bias: {best_b:.1f}")

    # --- 알고리즘 실행 ---
    ma_res, ma_time = apply_moving_average(raw)
    hd_res, hd_time = apply_hamming_correction(raw)
    nn_res, nn_time = apply_perceptron(raw, w_p, w_c, w_n, bias)

    # --- 결과 시각화 ---
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📊 필터링 결과 시각화")
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05)
        plots = [(raw, '원본', 'red'), (ma_res, '단순 평균', 'purple'), (hd_res, '해밍 거리', 'green'), (nn_res, '퍼셉트론', 'orange')]
        for i, (d, n, c) in enumerate(plots):
            fig.add_trace(go.Scatter(y=d, name=n, line=dict(color=c, width=2, shape='hv')), row=i+1, col=1)
        fig.update_layout(height=600, showlegend=False, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📈 통합 성능 리포트")
        r_raw = get_roughness(raw)
        
        perf_df = pd.DataFrame({
            "알고리즘": ["단순 평균", "해밍 거리", "퍼셉트론"],
            "유지율": [f"{get_stability(ma_res, raw):.1f}%", f"{get_stability(hd_res, raw):.1f}%", f"{get_stability(nn_res, raw):.1f}%"],
            "거칠기(점수)": [get_roughness(ma_res), get_roughness(hd_res), get_roughness(nn_res)],
            "노이즈제거율": [f"{((r_raw-get_roughness(ma_res))/r_raw)*100:.1f}%", f"{((r_raw-get_roughness(hd_res))/r_raw)*100:.1f}%", f"{((r_raw-get_roughness(nn_res))/r_raw)*100:.1f}%"],
            "속도(ms)": [f"{ma_time:.3f}", f"{hd_time:.3f}", f"{nn_time:.3f}"]
        })
        st.table(perf_df)
        
        st.metric("원본 데이터 거칠기", f"{r_raw} 점")
        st.info("💡 **유지율**은 원본과 얼마나 똑같은지를, **거칠기**는 신호가 얼마나 매끄러운지를 나타냅니다.")

    # --- 하단 상세 계산 ---
    st.divider()
    calc_idx = st.sidebar.slider("분석 인덱스", 1, len(raw)-2, 5)
    st.header(f"🧮 인덱스 {calc_idx} 상세 수치 대입")
    v = raw[calc_idx-1:calc_idx+2]
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### 1. 단순 평균")
        val = sum(v)/3
        st.latex(rf"\text{{Avg: }} {val:.2f} \to {'1' if val >= 0.5 else '0'}")
    with c2:
        st.markdown("### 2. 해밍 거리")
        d0 = sum([1 for x in v if x != 0])
        d1 = sum([1 for x in v if x != 1])
        st.write(f"000과의 거리: {d0}, 111과의 거리: {d1}")
        st.write(f"결과: **{'0' if d0 < d1 else '1'}**")
    with c3:
        st.markdown("### 3. 퍼셉트론")
        score = (v[0]*w_p + v[1]*w_c + v[2]*w_n) + bias
        st.latex(rf"Score: {score:.2f} \to {'1' if score > 0 else '0'}")

if __name__ == "__main__":
    main()
