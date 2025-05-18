import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import math

st.title("🏐 배구 스파이크 자세 분석기 (웹캠 촬영용)")

image_data = st.camera_input("📸 아래 버튼을 눌러 자세를 촬영하세요")

stop_frame = None
running = True

if image_data is not None:
    file_bytes = np.asarray(bytearray(image_data.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ✅ MediaPipe 포즈 초기화
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, model_complexity=1)
    results = pose.process(rgb)

    h, w, _ = frame.shape

    # ✅ HSV로 공 색상 탐지 (분홍색 기준)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([169, 101, 78])
    upper_hsv = np.array([179, 255, 255])
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    ball_detected = False
    ball_center = None

    if contours:
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 200:
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                center = (int(x), int(y))
                ball_center = center
                ball_detected = True
                cv2.circle(frame, center, int(radius), (255, 0, 255), 2)
                cv2.putText(frame, "Ball Detected", (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                break

    # ✅ 포즈 분석
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        def get_coords(idx): return [landmarks[idx].x * w, landmarks[idx].y * h]
        shoulder = get_coords(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
        elbow = get_coords(mp_pose.PoseLandmark.RIGHT_ELBOW.value)
        wrist = get_coords(mp_pose.PoseLandmark.RIGHT_WRIST.value)

        def calculate_angle(a, b, c):
            a, b, c = np.array(a), np.array(b), np.array(c)
            ab = a - b
            cb = c - b
            radians = np.arccos(np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb)))
            return np.degrees(radians)

        angle = calculate_angle(shoulder, elbow, wrist)

        # ✅ 피드백 출력
        if angle > 150:
            feedback = "✅ 아주 좋은 자세입니다!"
            color = (0, 255, 0)
        else:
            feedback = "⚠️ 공을 칠 때 팔꿈치를 더 펴보세요."
            color = (0, 0, 255)

        cv2.putText(frame, f"Angle: {int(angle)} deg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, feedback, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # ✅ 충돌 판단 추가
        finger_tip = wrist
        finger_tip_detected = True if finger_tip else False
        distance_threshold = 40  # 충돌 임계값 (픽셀 기준)

        if ball_detected and finger_tip_detected:
            distance = math.dist(ball_center, finger_tip)
            if distance < distance_threshold:
                stop_frame = frame.copy()
                running = False
    else:
        st.warning("사람을 인식하지 못했습니다. 자세히 촬영해주세요.")

    # ✅ 결과 출력
    if not running and stop_frame is not None:
        st.image(stop_frame, channels="BGR", caption="충돌 시점 캡처")
        st.success("손과 공이 접촉되었습니다!")
    else:
        st.image(frame, channels="BGR", caption="분석 결과")
