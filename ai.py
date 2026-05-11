import streamlit as st
import pandas as pd
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. 필터 알고리즘 정의 ---
def apply_moving_average(raw_list):
    start_time = time.perf_counter()
    # 리스트 컴프리헨션 시 범위 밖 인덱스 에러 방지 및 독립 리스트 생성
    res = []
    for i in range(len(raw_list)):
        if i == 0 or i == len(raw_list) - 1:
            res.append(raw_list[i])
        else:
            avg = (raw_list[i-1] + raw_list[i] + raw_list[i+1]) / 3.0
            res.append(1 if avg >= 0.5 else 0)
    return res, (time.perf_counter() - start_time) * 1000

def apply_hamming_correction(raw_list):
    start_time = time.perf_counter()
    res = []
    for i in range(len(raw_list)):
        if i == 0 or i == len(raw_list) - 1:
            res.append(raw_list[i])
        else:
            window = [raw_list[i-1], raw_list[i], raw_list[i+1]]
            d0 = sum([1 for v in window if v != 0])
            d1 = sum([1 for v in window if v != 1])
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

# --- 2. 분석 지표 계산 함수 (정밀도 향상) ---
def get_stability(res, raw):
    if not raw: return 0
    match = sum([1 for r, o in zip(res, raw) if r == o])
    return (match / len(raw)) * 100

def get_roughness(signal):
    if len(signal) < 2: return 0
    # 인접 데이터 간 변화 발생 횟수 측정
    return sum([1 for i in range(1, len(signal)) if signal[i] != signal[i-1]])

def main():
    st.set_page_config(layout="wide", page_title="PLC Filter Expert System")
    st.title("🏭 PLC 필터 정밀 진단 시스템")
    st.markdown("각 필터의 독립적인 성능 수치를 실시간으로 비교 분석합니다.")
    st.divider()

    # --- 사이드바 ---
    st.sidebar.header("📥 1. 데이터 입력")
    # 샘플 데이터를 조금 더 복잡하게 구성 (필터별 변별력 확보)
    default_data = "0000010000000101011111110111111111000000010000"
    custom_input = st.sidebar.text_area("센서 데이터 (0, 1)", value=default_data, height=100)
    raw = [int(c) for c in custom_input if c in '01']
    
    st.sidebar.divider()
    
    st.sidebar.header("⚙️ 2. 퍼셉트론 튜닝")
    w_p = st.sidebar.slider("W_past", 0.0, 2.0, 0.7, 0.1)
    w_c = st.sidebar.slider("W_curr", 0.0, 2.0, 1.2, 0.1)
    w_n = st.sidebar.slider("W_next", 0.0, 2.0, 0.7, 0.1)
    bias = st.sidebar.slider("Bias", -2.0, 0.0, -1.0, 0.1)

    if st.sidebar.button("🚀 최적 Bias 자동 탐색"):
        best_r = float('inf')
        best_b = bias
        for test_b in [round(x * -0.1, 2) for x in range(0, 21)]:
            res, _ = apply_perceptron(raw, w_p, w_c, w_n, test_b)
            r = get_roughness(res)
            if r < best_r:
                best_r = r
                best_b = test_b
        st.sidebar.success(f"추천 Bias: {best_b:.1f}")

    # --- 알고리즘 실행 (독립적 결과 생성) ---
    ma_res, ma_time = apply_moving_average(raw)
    hd_res, hd_time = apply_hamming_correction(raw)
    nn_res, nn_time = apply_perceptron(raw, w_p, w_c, w_n, bias)

    # --- 결과 시각화 ---
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📊 필터링 결과 비교")
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.07)
        
        plots = [
            (raw, '원본 입력', 'red'),
            (ma_res, '단순 평균', 'purple'),
            (hd_res, '해밍 거리', 'green'),
            (nn_res, '퍼셉트론', 'orange')
        ]
        
        calc_idx = st.slider("분석 인덱스", 1, len(raw)-2, 41) # 41번 근처가 노이즈 지점

        for i, (d, n, c) in enumerate(plots):
            fig.add_trace(go.Scatter(y=d, name=n, line=dict(color=c, width=2, shape='hv')), row=i+1, col=1)
            fig.add_vline(x=calc_idx, line_width=1, line_dash="dot", row=i+1, col=1)
            
        fig.update_layout(height=650, showlegend=False, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📈 통합 성능 리포트")
        r_raw = get_roughness(raw)
        
        # 필터 리스트와 이름 매핑
        results = [
            ("단순 평균", ma_res, ma_time),
            ("해밍 거리", hd_res, hd_time),
            ("퍼셉트론", nn_res, nn_time)
        ]
        
        perf_rows = []
        for name, res, t in results:
            r_val = get_roughness(res)
            # 노이즈 제거율 계산 보정
            improvement = ((r_raw - r_val) / r_raw * 100) if r_raw > 0 else 0
            perf_rows.append({
                "알고리즘": name,
                "유지율": f"{get_stability(res, raw):.1f}%",
                "거칠기(점수)": r_val,
                "노이즈제거율": f"{max(0, improvement):.1f}%",
                "연산속도(ms)": f"{t:.3f}"
            })
        
        st.table(pd.DataFrame(perf_rows))
        st.metric("원본 데이터 거칠기", f"{r_raw} 점")
        
        # 수치가 같게 나올 수 있는 이유 설명
        if len(set([row["거칠기(점수)"] for row in perf_rows])) == 1:
            st.info("💡 모든 필터의 거칠기가 같습니다. 입력 데이터의 노이즈가 단순하여 모든 필터가 동일하게 반응하고 있습니다. 사이드바에서 데이터를 수정해보세요.")

    # --- 하단 상세 계산 ---
    st.divider()
    st.header(f"🧮 인덱스 {calc_idx} 상세 수치 진단")
    v_p, v_c, v_n = raw[calc_idx-1], raw[calc_idx], raw[calc_idx+1]
    st.write(f"현재 윈도우: `[{v_p}, {v_c}, {v_n}]`")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 1. 단순 평균 (MA)")
        val = (v_p + v_c + v_n) / 3
        st.latex(rf"\text{{Avg: }} {val:.2f} \to {'1' if val >= 0.5 else '0'}")
    with c2:
        st.markdown("#### 2. 해밍 거리 (HD)")
        d0 = sum([1 for x in [v_p, v_c, v_n] if x != 0])
        d1 = sum([1 for x in [v_p, v_c, v_n] if x != 1])
        st.write(f"거리: d0={d0}, d1={d1} → **{'0' if d0 < d1 else '1'}**")
    with c3:
        st.markdown("#### 3. 퍼셉트론 (NN)")
        score = (v_p * w_p) + (v_c * w_c) + (v_n * w_n) + bias
        st.latex(rf"Score: {score:.2f} \to {'1' if score > 0 else '0'}")

if __name__ == "__main__":
    main()
