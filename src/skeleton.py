import cv2
import mediapipe as mp
import numpy as np

# --- CONFIGURATION ---
# Initialize MediaPipe Pose class
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def start_vision_system():
    # Setup the Pose function with confidence thresholds
    # min_detection_confidence: How sure the AI must be to say "There is a person here"
    # min_tracking_confidence: How sure it must be to keep tracking the same person
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        
        # Open the Webcam (0 is usually the default camera)
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Ignoring empty camera frame.")
                continue

            # 1. PRE-PROCESSING
            # Recolor Feed from BGR (Blue-Green-Red) to RGB for MediaPipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False # Performance optimization

            # 2. DETECTION (The Heavy Lifting)
            results = pose.process(image)

            # 3. POST-PROCESSING
            # Recolor back to BGR so OpenCV can draw on it
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # 4. EXTRACTING LANDMARKS
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates for the Right Wrist (Index 16 in MediaPipe)
                # We access x, y, z. They are normalized (0.0 to 1.0)
                wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                
                # Print coordinates to terminal (This is the raw data your AI will eventually learn!)
                print(f"Right Wrist - X: {wrist.x:.2f}, Y: {wrist.y:.2f}, Z: {wrist.z:.2f}")

                # Example Logic: Simple 'If' statement to test detection
                if wrist.y < 0.2: # If wrist is very high up in the frame
                    cv2.putText(image, 'HANDS UP!', (50,50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)

            except Exception as e:
                # If no person is detected, pass
                pass

            # 5. DRAW SKELETON ON SCREEN
            mp_drawing.draw_landmarks(
                image, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), # Joint color
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)  # Bone color
            )

            # Show the feed to the user
            cv2.imshow('Mediapipe Feed - Phase 1', image)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    start_vision_system()