import streamlit as st
import pandas as pd
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from functools import wraps
from typing import List, Tuple

# --- 0. 유틸리티: 연산 시간 측정 데코레이터 ---
# 이 데코레이터 덕분에 아래 필터 함수들에서 시간 측정 코드를 모두 지울 수 있습니다!
def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs) -> Tuple[List[int], float]:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_time_ms = (time.perf_counter() - start_time) * 1000
        return result, elapsed_time_ms
    return wrapper

# --- 1. 필터 알고리즘 정의 (핵심 로직만 남아 훨씬 깔끔해졌습니다) ---
@measure_time
def apply_moving_average(raw_list: List[int]) -> List[int]:
    res = []
    for i in range(len(raw_list)):
        if i == 0 or i == len(raw_list) - 1:
            res.append(raw_list[i])
        else:
            avg = (raw_list[i-1] + raw_list[i] + raw_list[i+1]) / 3.0
            res.append(1 if avg >= 0.5 else 0)
    return res

@measure_time
def apply_hamming_correction(raw_list: List[int]) -> List[int]:
    res = []
    for i in range(len(raw_list)):
        if i == 0 or i == len(raw_list) - 1:
            res.append(raw_list[i])
        else:
            window = [raw_list[i-1], raw_list[i], raw_list[i+1]]
            d0 = sum([1 for v in window if v != 0])
            d1 = sum([1 for v in window if v != 1])
            res.append(0 if d0 < d1 else 1)
    return res

@measure_time
def apply_perceptron(raw_list: List[int], w1: float, w2: float, w3: float, b: float) -> List[int]:
    res = []
    for i in range(len(raw_list)):
        if i == 0 or i == len(raw_list) - 1:
            res.append(raw_list[i])
        else:
            w_sum = (raw_list[i-1]*w1 + raw_list[i]*w2 + raw_list[i+1]*w3) + b
            res.append(1 if w_sum > 0 else 0)
    return res

# --- 2. 분석 지표 계산 함수 ---
def get_stability(res: List[int], raw: List[int]) -> float:
    match = sum([1 for r, o in zip(res, raw) if r == o])
    return (match / len(raw)) * 100

def get_roughness(signal: List[int]) -> int:
    return sum([1 for i in range(1, len(signal)) if signal[i] != signal[i-1]])

# --- 3. 메인 앱 실행부 ---
def main():
    st.set_page_config(layout="wide", page_title="PLC Filter Expert System", page_icon="🏭")
    
    with st.container():
        st.title("🏭 PLC 필터 정밀 진단 시스템")
        st.markdown("수학적 수식과 가중치 연산 과정을 실시간으로 추적하는 통합 엔지니어링 환경입니다.")
        st.divider()

    # --- 사이드바 ---
    with st.sidebar:
        st.header("📥 1. 데이터 입력")
        sample_data = "0000010000000101011111110111111111000000010000"
        custom_input = st.text_area("센서 데이터 (0, 1)", value=sample_data, height=100)
        raw = [int(c) for c in custom_input if c in '01']
        
        st.divider()
        
        st.header("⚙️ 2. 퍼셉트론 튜닝")
        w_p = st.slider("W_past (이전 값)", 0.0, 2.0, 0.7, 0.1, help="이전 데이터가 현재 판정에 미치는 영향력")
        w_c = st.slider("W_curr (현재 값)", 0.0, 2.0, 1.2, 0.1, help="현재 수집된 데이터의 신뢰도")
        w_n = st.slider("W_next (다음 값)", 0.0, 2.0, 0.7, 0.1, help="다음 데이터의 경향성 반영도")
        bias = st.slider("Bias (편향)", -2.0, 0.0, -1.0, 0.1, help="결과를 1로 판정하기 위한 허들값 (음수일수록 엄격)")

        if st.button("🚀 최적 Bias 자동 탐색", use_container_width=True):
            best_r = float('inf')
            best_b = bias
            for test_b in [round(x * -0.1, 2) for x in range(0, 21)]:
                res, _ = apply_perceptron(raw, w_p, w_c, w_n, test_b)
                r = get_roughness(res)
                if r < best_r:
                    best_r = r
                    best_b = test_b
            st.success(f"✅ 추천 Bias: {best_b:.1f}")

    # --- 알고리즘 실행 ---
    ma_res, ma_time = apply_moving_average(raw)
    hd_res, hd_time = apply_hamming_correction(raw)
    nn_res, nn_time = apply_perceptron(raw, w_p, w_c, w_n, bias)

    # --- 메인 화면 레이아웃 ---
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📊 실시간 필터링 비교")
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05)
        plots = [(raw, '원본 입력', 'red'), (ma_res, '단순 평균', 'purple'), (hd_res, '해밍 거리', 'green'), (nn_res, '퍼셉트론', 'orange')]
        
        calc_idx = st.slider("🔍 상세 분석할 인덱스를 선택하세요", 1, len(raw)-2, 5)

        for i, (d, n, c) in enumerate(plots):
            fig.add_trace(go.Scatter(y=d, name=n, line=dict(color=c, width=2, shape='hv')), row=i+1, col=1)
            fig.add_vline(x=calc_idx, line_width=1, line_dash="dot", row=i+1, col=1)
            
        fig.update_layout(height=600, showlegend=False, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📈 통합 성능 리포트")
        r_raw = get_roughness(raw)
        
        perf_df = pd.DataFrame({
            "알고리즘": ["단순 평균", "해밍 거리", "퍼셉트론"],
            "유지율": [f"{get_stability(ma_res, raw):.1f}%", f"{get_stability(hd_res, raw):.1f}%", f"{get_stability(nn_res, raw):.1f}%"],
            "거칠기(점수)": [get_roughness(ma_res), get_roughness(hd_res), get_roughness(nn_res)],
            "노이즈제거율": [
                f"{((r_raw - get_roughness(ma_res)) / r_raw * 100):.1f}%" if r_raw > 0 else "0%",
                f"{((r_raw - get_roughness(hd_res)) / r_raw * 100):.1f}%" if r_raw > 0 else "0%",
                f"{((r_raw - get_roughness(nn_res)) / r_raw * 100):.1f}%" if r_raw > 0 else "0%"
            ],
            "연산속도(ms)": [f"{ma_time:.3f}", f"{hd_time:.3f}", f"{nn_time:.3f}"]
        })
        st.table(perf_df)
        st.metric("원본 데이터 거칠기", f"{r_raw} 점")
        
        with st.expander("💡 지표 해석 가이드", expanded=True):
            st.caption("• **유지율:** 원본 데이터 특성을 얼마나 보존했는가?\n• **거칠기:** 선이 꺾이는 횟수 (낮을수록 안정적)\n• **제거율:** 불필요한 노이즈를 얼마나 깎아냈는가?")

    # --- 하단: 상세 계산 프로세스 ---
    st.divider()
    st.header(f"🧮 인덱스 {calc_idx}의 단계별 계산 과정")
    v_p, v_c, v_n = raw[calc_idx-1], raw[calc_idx], raw[calc_idx+1]
    st.markdown(f"**현재 관찰 윈도우:** 이전 : `{v_p}` , 현재 : `{v_c}` , 다음 : `{v_n}`")

    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("### 1. 단순 평균 (MA)")
        val = (v_p + v_c + v_n) / 3
        st.latex(rf"y_i = \text{{round}}\left(\frac{{x_{{i-1}} + x_i + x_{{i+1}}}}{{3}}\right)")
        st.write(f"계산: ({v_p} + {v_c} + {v_n}) / 3 = **{val:.2f}**")
        st.markdown(f"**결과: {'1' if val >= 0.5 else '0'}**")

    with c2:
        st.markdown("### 2. 해밍 거리 (HD)")
        st.latex(rf"D_H(x, pattern)")
        d0 = sum([1 for x in [v_p, v_c, v_n] if x != 0])
        d1 = sum([1 for x in [v_p, v_c, v_n] if x != 1])
        st.write(f"[0,0,0]과의 거리: **{d0}**")
        st.write(f"[1,1,1]과의 거리: **{d1}**")
        st.markdown(f"**결과: {'0' if d0 < d1 else '1'}**")

    with c3:
        st.markdown("### 3. 퍼셉트론 (NN)")
        st.latex(rf"\sigma(\sum W_i X_i + b)")
        w_sum = (v_p * w_p) + (v_c * w_c) + (v_n * w_n) + bias
        st.write(f"가중합: ({v_p}×{w_p:.1f}) + ({v_c}×{w_c:.1f}) + ({v_n}×{w_n:.1f}) + ({bias:.1f}) = **{w_sum:.2f}**")
        st.markdown(f"**결과: {'1' if w_sum > 0 else '0'}**")

if __name__ == "__main__":
    main()
