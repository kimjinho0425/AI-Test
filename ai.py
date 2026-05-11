import streamlit as st
import pandas as pd
import random

# [조건 6] 함수 정의: 실제 노이즈를 걸러내는 부울 로직(다수결 원칙) 필터
def apply_noise_filter(raw_list):
    # [조건 5] 리스트 활용: 필터링된 결과를 담을 빈 리스트
    filtered_list = [] 
    
    # [조건 4] 반복문 활용: 리스트의 처음부터 끝까지 순회
    for i in range(len(raw_list)):
        # 양 끝 데이터는 비교할 앞뒤 값이 부족하므로 그대로 유지
        if i == 0 or i == len(raw_list) - 1:
            filtered_list.append(raw_list[i])
        else:
            # [조건 1] 변수 사용: 앞, 현재, 뒤 3개의 신호를 추출
            prev_signal = raw_list[i-1]
            curr_signal = raw_list[i]
            next_signal = raw_list[i+1]
            
            # 다수결 부울 대수: 3개 중 1이 2개 이상이면 1, 아니면 0
            count_1 = prev_signal + curr_signal + next_signal
            
            # [조건 3] 조건문 활용: 1의 개수에 따라 필터링 결과 결정
            if count_1 >= 2:
                filtered_list.append(1)
            else:
                filtered_list.append(0)
                
    return filtered_list

def main():
    st.title("PLC 센서 노이즈 안정화 시뮬레이터 🏭")
    
    # [조건 7] 실제 문제 해결 기능 구현 설명
    st.markdown("""
    **문제 상황:** 공장 PLC 센서에 전자기 간섭으로 인해 짧은 노이즈(0이 1로, 1이 0으로 튀는 현상)가 발생하여 기계 오작동 유발.
    **해결 방안:** 부울 대수 기반의 다수결 논리(Majority Voting) 필터를 적용하여 응답 시간을 크게 지연시키지 않으면서 노이즈를 제거합니다.
    """)
    st.divider()

    # [조건 2] 사용자 입력 처리: Streamlit 슬라이더를 통해 노이즈 확률 입력받음
    st.sidebar.header("시뮬레이션 설정")
    noise_level = st.sidebar.slider("노이즈 발생 확률 (%)", min_value=0, max_value=50, value=15)

    # 이상적인 센서 신호 생성 (0과 1이 반복되는 펄스 신호)
    # [조건 5] 리스트 활용
    base_signal = [0]*15 + [1]*20 + [0]*15 + [1]*20 + [0]*15
    noisy_signal = []

    # 노이즈 주입
    # [조건 4] 반복문 활용
    for val in base_signal:
        # [조건 3] 조건문 활용: 설정한 확률에 따라 노이즈 발생
        if random.randint(1, 100) <= noise_level:
            noisy_signal.append(1 - val) # 0은 1로, 1은 0으로 반전
        else:
            noisy_signal.append(val)

    # [조건 6] 함수 호출: 필터링 함수 실행
    filtered_signal = apply_noise_filter(noisy_signal)

    # 결과를 Pandas 데이터프레임으로 묶어서 Streamlit 차트로 출력
    df = pd.DataFrame({
        "1. 원본 신호 (노이즈 포함)": noisy_signal,
        "2. 필터링된 신호 (안정화)": filtered_signal,
        "3. 이상적인 신호": base_signal
    })

    st.subheader("📊 센서 데이터 시각화")
    st.line_chart(df)
    
    st.success(f"현재 노이즈 확률 {noise_level}% 상황에서, 필터가 순간적인 스파이크를 성공적으로 무시하고 안정적인 출력을 유지하고 있습니다.")

# 파이썬 실행 시 메인 함수 호출
if __name__ == "__main__":
    main()
