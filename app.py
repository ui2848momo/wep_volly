
import cv2
import mediapipe as mp
import streamlit as st
import numpy as np
import math

st.set_page_config(layout="wide")
st.title("🏐 Volleyball Spike Analyzer (with Collision Detection & Pose Lines)")

# HSV 범위 설정 슬라이더
st.sidebar.header("🎨 Ball HSV Range (Pink)")
h_min = st.sidebar.slider("H Min", 0, 179, 169)
s_min = st.sidebar.slider("S Min", 0, 255, 101)
v_min = st.sidebar.slider("V Min", 0, 255, 78)
h_max = st.sidebar.slider("H Max", 0, 179, 179)
s_max = st.sidebar.slider("S Max", 0, 255, 255)
v_max = st.sidebar.slider("V Max", 0, 255, 255)

lower_pink = np.array([h_min, s_min, v_min])
upper_pink = np.array([h_max, s_max, v_max])

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

stop_frame = None
running = True
distance_threshold = 40  # 충돌 거리 임계값

# 웹캠 스트리밍 시작
cap = cv2.VideoCapture(0)
frame_placeholder = st.empty()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        st.warning("웹캠에서 프레임을 가져올 수 없습니다.")
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 포즈 추정
    results = pose.process(frame_rgb)

    # 손가락 끝과 공 중심 초기화
    finger_tip = None
    ball_center = None
    ball_detected = False
    finger_tip_detected = False

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
        )

        # 어깨, 팔꿈치, 손목으로 각도 계산
        landmarks = results.pose_landmarks.landmark
        shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        finger_tip = (int(wrist.x * frame.shape[1]), int(wrist.y * frame.shape[0]))
        finger_tip_detected = True

        def calc_angle(a, b, c):
            a = np.array(a)
            b = np.array(b)
            c = np.array(c)
            radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
            angle = np.abs(radians * 180.0 / np.pi)
            return 360 - angle if angle > 180.0 else angle

        angle = calc_angle(
            (shoulder.x, shoulder.y),
            (elbow.x, elbow.y),
            (wrist.x, wrist.y)
        )

        cv2.putText(
            frame, f"Angle: {int(angle)} deg", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
        )

    # 공 탐지 (HSV 마스크)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_pink, upper_pink)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(largest)
        if radius > 10:
            ball_center = (int(x), int(y))
            cv2.circle(frame, ball_center, int(radius), (255, 0, 255), 2)
            cv2.putText(frame, "Ball Detected", (int(x), int(y)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            ball_detected = True

    # 손과 공 거리 계산 -> 충돌 여부
    if ball_detected and finger_tip_detected:
        distance = math.dist(ball_center, finger_tip)
        if distance < distance_threshold:
            stop_frame = frame.copy()
            running = False

    # 결과 출력
    if not running and stop_frame is not None:
        frame_placeholder.image(stop_frame, channels="BGR", caption="충돌 시점 캡처")
        st.success("손과 공이 접촉되었습니다!")
        break
    else:
        frame_placeholder.image(frame, channels="BGR")

cap.release()
