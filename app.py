import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import math

st.set_page_config(layout="wide")
st.title("🏐 Volleyball Spike Analyzer (with Elbow Feedback)")

# 사이드바 HSV 슬라이더
st.sidebar.header("🎨 Ball HSV Range (Pink)")
h_min = st.sidebar.slider("H Min", 0, 179, 169)
s_min = st.sidebar.slider("S Min", 0, 255, 101)
v_min = st.sidebar.slider("V Min", 0, 255, 78)
h_max = st.sidebar.slider("H Max", 0, 179, 179)
s_max = st.sidebar.slider("S Max", 0, 255, 255)
v_max = st.sidebar.slider("V Max", 0, 255, 255)

lower_hsv = np.array([h_min, s_min, v_min])
upper_hsv = np.array([h_max, s_max, v_max])

# MediaPipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# 웹캠 캡처
cap = cv2.VideoCapture(0)
stframe = st.empty()

# 충돌 감지를 위한 설정
distance_threshold = 40
stop_frame = None
running = True

while running:
    ret, frame = cap.read()
    if not ret:
        st.error("웹캠에서 프레임을 가져올 수 없습니다.")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    height, width, _ = frame.shape
    elbow_angle = None
    ball_center = None
    finger_tip = None
    ball_detected = False
    finger_tip_detected = False

    # 포즈 추정 및 각도 계산
    if result.pose_landmarks:
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        landmarks = result.pose_landmarks.landmark
        try:
            shoulder = [int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * width),
                        int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * height)]
            elbow = [int(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x * width),
                     int(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y * height)]
            wrist = [int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x * width),
                     int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y * height)]
            finger_tip = [int(landmarks[mp_pose.PoseLandmark.RIGHT_INDEX].x * width),
                          int(landmarks[mp_pose.PoseLandmark.RIGHT_INDEX].y * height)]
            finger_tip_detected = True

            # 각도 계산
            a = np.array(shoulder)
            b = np.array(elbow)
            c = np.array(wrist)

            ba = a - b
            bc = c - b
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
            elbow_angle = np.degrees(angle)

            # 피드백 출력
            feedback = "좋은 스윙!" if elbow_angle > 120 else "팔을 더 펴세요!"
            color = (0, 255, 0) if elbow_angle > 120 else (0, 0, 255)
            cv2.putText(frame, f"Angle: {int(elbow_angle)} deg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, feedback, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        except:
            pass

    # 공 검출 (HSV 마스크)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 300:
            (x, y), radius = cv2.minEnclosingCircle(largest)
            center = (int(x), int(y))
            ball_center = center
            ball_detected = True
            cv2.circle(frame, center, int(radius), (255, 0, 255), 2)
            cv2.putText(frame, "Ball Detected", (center[0]+10, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    # 충돌 감지
    if ball_detected and finger_tip_detected:
        distance = math.dist(ball_center, finger_tip)
        if distance < distance_threshold:
            stop_frame = frame.copy()
            running = False

    stframe.image(frame, channels="BGR", caption="실시간 분석", use_column_width=True)

# 충돌 시점 정지 화면
cap.release()
if stop_frame is not None:
    st.image(stop_frame, channels="BGR", caption="충돌 시점 캡처", use_column_width=True)
    st.success("손과 공이 접촉되었습니다!")
