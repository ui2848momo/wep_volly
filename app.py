import streamlit as st
import cv2
import numpy as np
import mediapipe as mp

st.title("🏐 배구 스파이크 자세 분석기 (웹캠 촬영용)")

image_data = st.camera_input("📸 아래 버튼을 눌러 자세를 촬영하세요")

if image_data is not None:
    file_bytes = np.asarray(bytearray(image_data.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)

    # MediaPipe 포즈 초기화
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        h, w, _ = frame.shape

        # 팔꿈치 각도 계산 (오른팔 기준)
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

        # 피드백 메시지
        if angle > 150:
            feedback = "✅ 아주 좋은 자세입니다!"
            color = (0, 255, 0)
        else:
            feedback = "⚠️ 공을 칠 때 팔꿈치를 더 펴보세요."
            color = (0, 0, 255)

        # 결과 출력
        cv2.putText(frame, f"Angle: {int(angle)} deg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, feedback, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        st.image(frame, channels="BGR", caption="분석 결과")

    else:
        st.warning("사람을 인식하지 못했습니다. 자세히 촬영해주세요.")
