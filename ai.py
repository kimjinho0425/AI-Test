import streamlit as st
import pandas as pd
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. 필터 알고리즘 정의 ---
def apply_moving_average(raw_list):
    start_time = time.perf_counter()
    res = [raw_list[i] if i == 0 or i == len(raw_list)-1 else (1 if (sum(raw_list[i-1:i+2])/3.0) >= 0.5 else 0) for i in range(len(raw_list))]
    return res, (time.perf_counter() - start_time) * 1000

def apply_median_filter(raw_list):
    start_time = time.perf_counter()
    res = []
    for i in range(len(raw_list)):
        if i == 0 or i == len(raw_list) - 1: res.append(raw_list[i])
        else:
            # 3개 데이터 중 중간값 선택 (튀는 노이즈 제거에 특화)
            window = sorted([raw_list[i-1], raw_list[i], raw_list[i+1]])
            res.append(window[1])
    return res, (time.perf_counter() - start_time) * 1000

def apply_perceptron(raw_list, w1, w2, w3, b):
    start_time = time.perf_counter()
    res = []
    for i in range(len(raw_list)):
        if i == 0 or i == len(raw_list) - 1: res.append(raw_list[i])
        else:
            w_sum = (raw_list[i-1]*w1 + raw_list[i]*w2 + raw_list[i+1]*w3) + b
            res.append(1 if w_sum > 0 else 0)
    return res, (time.perf_counter() - start_time) * 1000

# --- 2. 심화 지표: 신호 거칠기(Roughness) 계산 함수 ---
def get_roughness(signal):
    # 인접 데이터 간의 차이 절댓값 합 (변화가 잦을수록 높은 점수)
    return sum([abs(signal[i] - signal[i-1]) for i in range(1, len(signal))])

def main():
    st.set_page_config(layout="wide", page_title="PLC Signal Expert")
    st.title("🔬 PLC 신호 정밀 진단 및 자동 최적화 시스템")
    st.markdown("신호의 수학적 거칠기를 분석하고, 가장 매끄러운 신호를 만드는 최적의 가중치를 도출합니다.")
    st.divider()

    # --- 사이드바: 입력 및 최적화 ---
    st.sidebar.header("📥 1. 데이터 입력")
    sample_data = "0000010000000101011111110111111111000000010000"
    custom_input = st.sidebar.text_area("센서 데이터 (0, 1)", value=sample_data, height=100)
    raw = [int(c) for c in custom_input if c in '01']
    
    st.sidebar.divider()
    st.sidebar.header("⚙️ 2. 퍼셉트론 파라미터")
    w_p = st.sidebar.slider("이전 값 가중치 (W_past)", 0.0, 2.0, 0.7, 0.1)
    w_c = st.sidebar.slider("현재 값 가중치 (W_curr)", 0.0, 2.0, 1.2, 0.1)
    w_n = st.sidebar.slider("다음 값 가중치 (W_next)", 0.0, 2.0, 0.7, 0.1)
    bias = st.sidebar.slider("판단 편향 (Bias)", -2.0, 0.0, -1.0, 0.1)

    # --- [심화] 자동 최적화 버튼 ---
    if st.sidebar.button("🚀 최적 Bias 자동 탐색"):
        best_r = float('inf')
        best_b = bias
        # Bias를 -2.0부터 0.0까지 테스트하여 거칠기가 가장 낮은 지점 탐색
        for test_b in [x * -0.1 for x in range(0, 21)]:
            res, _ = apply_perceptron(raw, w_p, w_c, w_n, test_b)
            r = get_roughness(res)
            if r < best_r:
                best_r = r
                best_b = test_b
        st.sidebar.success(f"추천 Bias 발견: {best_b:.1f}")
        st.sidebar.info(f"이 값을 적용하면 신호가 가장 매끄러워집니다.")

    # --- 알고리즘 실행 ---
    ma_res, ma_time = apply_moving_average(raw)
    md_res, md_time = apply_median_filter(raw)
    nn_res, nn_time = apply_perceptron(raw, w_p, w_c, w_n, bias)

    # --- 메인 화면 레이아웃 ---
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📊 실시간 필터링 비교")
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05)
        titles = ["원본 입력", "단순 평균", "중간값 필터(신규)", "퍼셉트론(Tuned)"]
        signals = [raw, ma_res, md_res, nn_res]
        colors = ['red', 'purple', 'blue', 'orange']

        for i, (sig, clr, ttl) in enumerate(zip(signals, colors, titles)):
            fig.add_trace(go.Scatter(y=sig, line=dict(color=clr, width=2, shape='hv'), name=ttl), row=i+1, col=1)
        
        fig.update_layout(height=600, margin=dict(t=20, b=20), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📋 신호 정밀 진단서")
        r_raw = get_roughness(raw)
        
        # 성능 지표 데이터프레임
        perf_data = []
        for name, sig, t in [("이동 평균", ma_res, ma_time), 
                             ("중간값 필터", md_res, md_time), 
                             ("퍼셉트론", nn_res, nn_time)]:
            r = get_roughness(sig)
            improvement = ((r_raw - r) / r_raw * 100) if r_raw > 0 else 0
            perf_data.append({
                "필터 종류": name,
                "거칠기 점수": r,
                "노이즈 제거율": f"{improvement:.1f}%",
                "속도(ms)": f"{t:.3f}"
            })
        
        st.table(pd.DataFrame(perf_data))
        st.metric("원본 신호 거칠기", f"{r_raw} 점")
        st.caption("※ 거칠기 점수가 낮을수록 신호가 안정적임을 의미합니다.")

    # --- 하단: 상세 계산 프로세스 ---
    st.divider()
    calc_idx = st.sidebar.slider("분석 인덱스 선택", 1, len(raw)-2, 5)
    st.header(f"🧮 인덱스 {calc_idx}의 수치 진단")
    v_p, v_c, v_n = raw[calc_idx-1], raw[calc_idx], raw[calc_idx+1]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### 1. 단순 평균")
        val = (v_p + v_c + v_n) / 3
        st.latex(rf"\frac{{{v_p} + {v_c} + {v_n}}}{{3}} = {val:.2f} \to {'1' if val >= 0.5 else '0'}")
    with c2:
        st.markdown("### 2. 중간값 필터")
        sorted_v = sorted([v_p, v_c, v_n])
        st.write(f"정렬: {sorted_v} → 중앙값: **{sorted_v[1]}**")
    with c3:
        st.markdown("### 3. 퍼셉트론")
        w_sum = (v_p * w_p) + (v_c * w_c) + (v_n * w_n) + bias
        st.latex(rf"Score: {w_sum:.2f} \to {'1' if w_sum > 0 else '0'}")

if __name__ == "__main__":
    main()
