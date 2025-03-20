import cv2
import mediapipe as mp
import os
import time

# Suppress TensorFlow and MediaPipe warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_hands_module = mp.solutions.hands

# Video capture setup
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Video writer setup
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('white_hand_detection.avi', fourcc, 20.0, (frame_width, frame_height))

# FPS calculation variables
prev_time = 0

# Hand tracking with MediaPipe
with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip frame horizontally for a natural webcam view
        frame = cv2.flip(frame, 1)

        # Convert frame to RGB for processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame with MediaPipe
        results = hands.process(rgb_frame)

        # FPS calculation
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        # If hands detected, process them
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Identify left or right hand
                handedness = results.multi_handedness[idx].classification[0].label  # 'Left' or 'Right'
                
                # Define text position
                text_position = (int(hand_landmarks.landmark[0].x * frame_width), 
                                 int(hand_landmarks.landmark[0].y * frame_height) - 10)

                # Display Left or Right Hand Text
                cv2.putText(frame, handedness, text_position, 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)  # White text
                
                # Draw hand landmarks in white
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=3),  # White color
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)   # White connections
                )

        # Display FPS on the frame
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)  # White FPS text

        # Show the output frame
        cv2.imshow('White Hand Detection with Left & Right Identification', frame)

        # Write frame to video file
        out.write(frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources properly
cap.release()
out.release()
cv2.destroyAllWindows()
