import cv2
import mediapipe as mp
import streamlit as st
import numpy as np

# MediaPipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Streamlit 설정
st.set_page_config(layout="wide")
st.title("🏐 Volleyball Spike Analyzer (Webcam + Auto Stop)")
st.sidebar.header("🎨 Ball HSV Range (Pink)")
h_min = st.sidebar.slider("H Min", 0, 179, 169)
s_min = st.sidebar.slider("S Min", 0, 255, 101)
v_min = st.sidebar.slider("V Min", 0, 255, 78)
h_max = st.sidebar.slider("H Max", 0, 179, 179)
s_max = st.sidebar.slider("S Max", 0, 255, 255)
v_max = st.sidebar.slider("V Max", 0, 255, 255)

# HSV 범위
lower_hsv = np.array([h_min, s_min, v_min])
upper_hsv = np.array([h_max, s_max, v_max])

# 유틸 함수: 손가락 끝 좌표 추출
def get_index_finger_tip(results, image_shape):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = image_shape
            x = int(hand_landmarks.landmark[8].x * w)
            y = int(hand_landmarks.landmark[8].y * h)
            return (x, y)
    return None

# 유틸 함수: 공 중심 좌표 추출
def get_ball_center(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 100:
            M = cv2.moments(largest)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return (cx, cy)
    return None

# 스트리밍 실행
cap = cv2.VideoCapture(0)
st_frame = st.empty()
captured = False

while cap.isOpened() and not captured:
    ret, frame = cap.read()
    if not ret:
        st.error("❌ 웹캠에서 영상을 불러올 수 없습니다.")
        break

    # 좌우 반전 및 전처리
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 포즈/손/색상 추론
    pose_results = pose.process(rgb)
    hand_results = hands.process(rgb)

    # 공 마스크
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # 포즈 그리기
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # 손가락 위치
    fingertip = get_index_finger_tip(hand_results, frame.shape)
    if fingertip:
        cv2.circle(frame, fingertip, 5, (255, 0, 255), -1)

    # 공 위치
    ball_center = get_ball_center(mask)
    if ball_center:
        cv2.circle(frame, ball_center, 10, (0, 255, 255), 2)

    # 충돌 판단
    if fingertip and ball_center:
        distance = np.linalg.norm(np.array(fingertip) - np.array(ball_center))
        if distance < 40:
            cv2.putText(frame, "Hit!", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            captured = True
            st.success("🎉 충돌 감지! 프레임 캡처됨.")
            st.image(frame, channels="BGR", caption="📸 충돌 순간")
            break

    # 실시간 스트리밍 출력
    st_frame.image(frame, channels="BGR")

cap.release()
