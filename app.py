
import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
import math

st.set_page_config(layout="wide")
st.title("🏐 Volleyball Spike Analyzer (실시간 웹캠 + 충돌 멈춤 + 포즈 추적)")

# HSV 슬라이더 설정
st.sidebar.header("🎨 공 색상 HSV 범위")
h_min = st.sidebar.slider("H min", 0, 179, 169)
s_min = st.sidebar.slider("S min", 0, 255, 101)
v_min = st.sidebar.slider("V min", 0, 255, 78)
h_max = st.sidebar.slider("H max", 0, 179, 179)
s_max = st.sidebar.slider("S max", 0, 255, 255)
v_max = st.sidebar.slider("V max", 0, 255, 255)

lower_hsv = np.array([h_min, s_min, v_min])
upper_hsv = np.array([h_max, s_max, v_max])

# MediaPipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
stframe = st.empty()

distance_threshold = 40
stop_frame = None
running = True

while cap.isOpened() and running:
    ret, frame = cap.read()
    if not ret:
        st.warning("웹캠 프레임을 불러올 수 없습니다.")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    height, width, _ = frame.shape
    ball_center = None
    finger_tip = None
    ball_detected = False
    finger_tip_detected = False

    # 공 탐지
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 300:
            (x, y), radius = cv2.minEnclosingCircle(largest)
            ball_center = (int(x), int(y))
            ball_detected = True
            cv2.circle(frame, ball_center, int(radius), (255, 0, 255), 2)
            cv2.putText(frame, "Ball", (int(x), int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    # 포즈 추정 + 손가락 위치
    if result.pose_landmarks:
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        lm = result.pose_landmarks.landmark
        index = lm[mp_pose.PoseLandmark.RIGHT_INDEX]
        finger_tip = (int(index.x * width), int(index.y * height))
        finger_tip_detected = True
        cv2.circle(frame, finger_tip, 6, (0, 255, 0), -1)

    # 충돌 판단
    if ball_detected and finger_tip_detected:
        distance = math.dist(ball_center, finger_tip)
        if distance < distance_threshold:
            stop_frame = frame.copy()
            running = False

    stframe.image(frame, channels="BGR", use_column_width=True)

cap.release()
if stop_frame is not None:
    st.image(stop_frame, channels="BGR", caption="📸 충돌 시점 캡처", use_column_width=True)
    st.success("손과 공이 충돌하여 영상이 자동으로 멈췄습니다!")
