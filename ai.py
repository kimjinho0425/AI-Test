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

def apply_hamming_correction(raw_list):
    start_time = time.perf_counter()
    res = []
    for i in range(len(raw_list)):
        if i == 0 or i == len(raw_list) - 1:
            res.append(raw_list[i])
        else:
            window = [raw_list[i-1], raw_list[i], raw_list[i+1]]
            d0 = sum([1 for v in window if v != 0]) # 000 패턴과의 거리
            d1 = sum([1 for v in window if v != 1]) # 111 패턴과의 거리
            res.append(0 if d0 < d1 else 1)
    return res, (time.perf_counter() - start_time) * 1000

def apply_perceptron(raw_list, w1, w2, w3, b):
    start_time = time.perf_counter()
    res = []
    for i in range(len(raw_list)):
        if i == 0 or i == len(raw_list) - 1:
            res.append(raw_list[i])
        else:
            # 설정된 가중치와 편향 적용
            w_sum = (raw_list[i-1]*w1 + raw_list[i]*w2 + raw_list[i+1]*w3) + b
            res.append(1 if w_sum > 0 else 0)
    return res, (time.perf_counter() - start_time) * 1000

# --- 2. 분석 지표 계산 함수 ---
def get_stability(res, raw):
    """원본 데이터와의 일치율(유지율) 계산"""
    match = sum([1 for r, o in zip(res, raw) if r == o])
    return (match / len(raw)) * 100

def get_roughness(signal):
    """신호 거칠기(변화량 합계) 계산"""
    return sum([abs(signal[i] - signal[i-1]) for i in range(1, len(signal))])

def main():
    st.set_page_config(layout="wide", page_title="PLC Filter Expert System")
    st.title("🏭 PLC 필터 정밀 진단 시스템 (Full Stack)")
    st.markdown("기본 필터링부터 AI 튜닝, 수학적 거칠기 분석까지 통합된 엔지니어링 도구입니다.")
    st.divider()

    # --- 사이드바: 데이터 입력 및 파라미터 튜닝 ---
    st.sidebar.header("📥 1. 데이터 입력")
    sample_data = "0000010000000101011111110111111111000000010000"
    custom_input = st.sidebar.text_area("센서 데이터 (0, 1)", value=sample_data, height=100)
    raw = [int(c) for c in custom_input if c in '01']
    
    st.sidebar.divider()
    
    st.sidebar.header("⚙️ 2. 퍼셉트론 파라미터 튜닝")
    st.sidebar.caption("가중치와 편향을 조절하여 필터의 성격을 결정합니다.")
    w_p = st.sidebar.slider("W_past (이전 값)", 0.0, 2.0, 0.7, 0.1)
    w_c = st.sidebar.slider("W_curr (현재 값)", 0.0, 2.0, 1.2, 0.1)
    w_n = st.sidebar.slider("W_next (다음 값)", 0.0, 2.0, 0.7, 0.1)
    bias = st.sidebar.slider("Bias (편향)", -2.0, 0.0, -1.0, 0.1)

    st.sidebar.divider()
    
    # [심화 기능] 자동 최적화 버튼
    if st.sidebar.button("🚀 최적 Bias 자동 탐색"):
        best_r = float('inf')
        best_b = bias
        # 거칠기를 최소화하는 Bias 지점 탐색 (AI 학습의 기초)
        for test_b in [x * -0.1 for x in range(0, 21)]:
            res, _ = apply_perceptron(raw, w_p, w_c, w_n, test_b)
            r = get_roughness(res)
            if r < best_r:
                best_r = r
                best_b = test_b
        st.sidebar.success(f"추천 Bias 발견: {best_b:.1f}")
        st.sidebar.info("해당 값으로 슬라이더를 조정하면 신호가 가장 매끄러워집니다.")

    # --- 알고리즘 실행 ---
    ma_res, ma_time = apply_moving_average(raw)
    hd_res, hd_time = apply_hamming_correction(raw)
    nn_res, nn_time = apply_perceptron(raw, w_p, w_c, w_n, bias)

    # --- 메인 화면 레이아웃 ---
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📊 실시간 필터링 비교 시각화")
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05)
        
        # 그래프 데이터 설정
        plots = [
            (raw, '원본 입력(Raw)', 'red'),
            (ma_res, '단순 평균(MA)', 'purple'),
            (hd_res, '해밍 거리(HD)', 'green'),
            (nn_res, '퍼셉트론(NN)', 'orange')
        ]
        
        # 분석 포인트 인덱스 표시를 위한 슬라이더
        calc_idx = st.slider("분석 타겟 인덱스 선택", 1, len(raw)-2, 5)

        for i, (d, n, c) in enumerate(plots):
            fig.add_trace(go.Scatter(y=d, name=n, line=dict(color=c, width=2, shape='hv')), row=i+1, col=1)
            # 분석 지점 수직선 표시
            fig.add_vline(x=calc_idx, line_width=1, line_dash="dot", row=i+1, col=1)
            
        fig.update_layout(height=600, showlegend=False, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📈 통합 성능 리포트")
        r_raw = get_roughness(raw)
        
        # 성능 지표 통합 테이블 생성
        perf_df = pd.DataFrame({
            "알고리즘": ["단순 평균", "해밍 거리", "퍼셉트론"],
            "유지율": [f"{get_stability(ma_res, raw):.1f}%", f"{get_stability(hd_res, raw):.1f}%", f"{get_stability(nn_res, raw):.1f}%"],
            "거칠기(점수)": [get_roughness(ma_res), get_roughness(hd_res), get_roughness(nn_res)],
            "노이즈제거율": [
                f"{((r_raw - get_roughness(ma_res)) / r_raw * 100):.1f}%",
                f"{((r_raw - get_roughness(hd_res)) / r_raw * 100):.1f}%",
                f"{((r_raw - get_roughness(nn_res)) / r_raw * 100):.1f}%"
            ],
            "연산속도(ms)": [f"{ma_time:.3f}", f"{hd_time:.3f}", f"{nn_time:.3f}"]
        })
        
        st.table(perf_df)
        
        # 원본 상태 요약
        st.metric("원본 데이터 거칠기", f"{r_raw} 점")
        st.warning("**엔지니어 가이드**")
        st.write("- **유지율:** 원본 데이터의 특성을 얼마나 보존하는가?")
        st.write("- **거칠기:** 신호가 얼마나 매끄럽게 정제되었는가?")
        st.write("- **제거율:** 원본 노이즈를 수학적으로 얼마나 삭감했는가?")

    # --- 하단: 상세 계산 프로세스 (공식 및 수치 대입) ---
    st.divider()
    st.header(f"🧮 인덱스 {calc_idx}의 실시간 수학적 진단")
    v_p, v_c, v_n = raw[calc_idx-1], raw[calc_idx], raw[calc_idx+1]
    st.info(f"현재 분석 윈도우: `이전:{v_p}` , `현재:{v_c}` , `다음:{v_n}`")

    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("### 1. 단순 평균 (MA)")
        val = (v_p + v_c + v_n) / 3
        st.latex(rf"\text{{Result}} = \text{{round}}\left(\frac{{{v_p} + {v_c} + {v_n}}}{{3}}\right) = {val:.2f}")
        st.markdown(f"판정: **{'1' if val >= 0.5 else '0'}**")

    with c2:
        st.markdown("### 2. 해밍 거리 (HD)")
        d0 = sum([1 for x in [v_p, v_c, v_n] if x != 0])
        d1 = sum([1 for x in [v_p, v_c, v_n] if x != 1])
        st.write(f"패턴 [0,0,0]과의 거리: **{d0}**")
        st.write(f"패턴 [1,1,1]과의 거리: **{d1}**")
        st.markdown(f"판정: **{'0' if d0 < d1 else '1'}** (가까운 쪽)")

    with c3:
        st.markdown("### 3. 퍼셉트론 (NN)")
        score = (v_p * w_p) + (v_c * w_c) + (v_n * w_n) + bias
        st.latex(rf"({v_p} \cdot {w_p}) + ({v_c} \cdot {w_c}) + ({v_n} \cdot {w_n}) + ({bias}) = {score:.2f}")
        st.markdown(f"판정: **{'1' if score > 0 else '0'}** (활성화 함수)")

if __name__ == "__main__":
    main()
