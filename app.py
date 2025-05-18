import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
from PIL import Image

st.set_page_config(layout="wide")
st.title("🏐 Volleyball Spike Analyzer (Auto Capture with Feedback)")

# HSV 색상 범위 슬라이더
st.sidebar.subheader("🎨 Ball HSV Range (Pink)")
h_min = st.sidebar.slider("H Min", 0, 179, 169)
s_min = st.sidebar.slider("S Min", 0, 255, 101)
v_min = st.sidebar.slider("V Min", 0, 255, 78)
h_max = st.sidebar.slider("H Max", 0, 179, 179)
s_max = st.sidebar.slider("S Max", 0, 255, 255)
v_max = st.sidebar.slider("V Max", 0, 255, 255)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

FRAME_WINDOW = st.image([])

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arccos(np.clip(np.dot(b-a, c-b) / 
                                (np.linalg.norm(b-a) * np.linalg.norm(c-b)), -1.0, 1.0))
    return np.degrees(radians)

cap = cv2.VideoCapture(0)
captured = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.warning("웹캠에서 프레임을 가져올 수 없습니다.")
        break

    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    # 공 탐지 (색상 기반)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (h_min, s_min, v_min), (h_max, s_max, v_max))
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ball_center = None
    if contours:
        largest = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(largest)
        if radius > 5:
            ball_center = (int(x), int(y))
            cv2.circle(frame, ball_center, int(radius), (255, 0, 255), 2)

    # 포즈 추정
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        # 좌우 중 하나 기준 (여기선 오른쪽)
        shoulder = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                    lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        elbow = [lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                 lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        wrist = [lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                 lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        h, w, _ = frame.shape
        points = [tuple(int(i * j) for i, j in zip(p, (w, h))) for p in [shoulder, elbow, wrist]]
        angle = calculate_angle(*points)
        cv2.putText(frame, f"Angle: {int(angle)} deg", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 손목 위치와 공 중심 간 거리 계산
        if ball_center:
            hand = points[2]
            dist = np.linalg.norm(np.array(hand) - np.array(ball_center))
            if dist < 50 and not captured:
                cv2.putText(frame, "💥 HIT!", (ball_center[0], ball_center[1]-10),
                            cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
                img_result = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                st.success("🎉 공과 손이 충돌했습니다. 분석 결과 캡처됨!")
                st.image(img_result, caption="📸 충돌 시점", use_column_width=True)
                captured = True
                break  # 정지

    FRAME_WINDOW.image(frame, channels="BGR")
else:
    st.warning("웹캠 연결을 확인해주세요.")

cap.release()
