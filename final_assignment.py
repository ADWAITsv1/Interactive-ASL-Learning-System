import cv2
import mediapipe as mp
import numpy as np
import serial
import time
import json # For saving/loading data
import os   # For path operations

# --- Configuration ---
# Path to your downloaded ASL tutorial video
VIDEO_PATH = '/Users/ady/Desktop/My_projects/Takfujisenseiproj/asl_tutorial.mp4'
# Serial port for your Arduino. IMPORTANT: Change this to your Arduino's port!
SERIAL_PORT = '/dev/cu.usbmodem201912341'
BAUD_RATE = 9600

# Video segment for learning and quizzing (adjusted for the new video: A-Z)
# These are the overall start/end times for the learning segment, not individual quiz points.
START_TIME_SEC = 11 # 'A' starts around 0:11 in the new video
END_TIME_SEC = 175.0   # Extended end time to cover all new signs up to 'Z' (Z is at 117.0s, giving buffer)

# This now acts as a delay AFTER a quiz is matched/timed out, before the NEXT sign is considered.
QUIZ_INTERVAL_SEC = 3 # Time to wait after a quiz before moving to the next sign

# Maximum time (in seconds) for the user to match the sign
QUIZ_DURATION_SEC = 5

# Similarity threshold for a "correct" match.
# IMPORTANT: This is Euclidean distance. Lower value = closer match.
MATCH_THRESHOLD = 1.5 # Increased threshold to be more forgiving

# Duration (in seconds) to search for a hand around the specified timestamp during reference loading
REFERENCE_LOAD_SEARCH_WINDOW_SEC = 2.0 # Increased search window to 2.0 seconds

# --- Reference Signs Configuration ---
# Define the signs to be quizzed and their exact timestamp (in seconds) in the video.
# These timestamps are based on the image you provided.
REFERENCE_SIGNS_CONFIG = {
    'A': 15.0,
    'B': 19.0,
    'C': 26.0,
    'D': 33.0,
    'E': 39.0,
    'F': 46.0,
    'G': 51.0,
    'H': 56.0,
    'I': 62.0,
    'J': 69.0, # Using 1:09 from your note
    'K': 80.0, # 1:20
    'L': 86.0, # 1:26
    'M': 93.0, # 1:33
    'N': 97.0, # 1:37
    'O': 103.0, # 1:43
    'P': 110.0, # 1:50
    'Q': 116.0, # 1:56
    'R': 122.0, # 2:02
    'S': 128.0, # 2:08
    'T': 133.0, # 2:13
    'U': 139.0, # 2:19
    'V': 145.0, # 2:25
    'W': 151.0, # 2:31
    'X': 155.0, # 2:35
    'Y': 161.0, # 2:41
    'Z': 171.0  # 2:51
}

# --- MediaPipe Setup ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize MediaPipe Hands for video processing (static image mode for single frame processing)
hands_video = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.1, min_tracking_confidence=0.5)
hands_webcam = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- Serial Communication Setup ---
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2) # Give some time for the serial connection to establish
    print(f"Serial connection established on {SERIAL_PORT}")
except serial.SerialException as e:
    print(f"Error opening serial port {SERIAL_PORT}: {e}")
    print("Please ensure Arduino is connected and the correct port is selected.")
    print("Exiting...")
    exit()

# --- Helper Functions ---

def send_serial_command(command_char, message_for_lcd=""):
    """
    Sends a command in the format 'G:CORRECT\n' or 'R:INCORRECT\n' to Arduino.
    LCD messages should be concise (e.g., "CORRECT", "NO HAND CAM").
    """
    try:
        full_message = f"{command_char}:{message_for_lcd}\n"
        ser.write(full_message.encode())
        # print(f"Sent to Arduino: {full_message.strip()}")  # Optional debug
    except serial.SerialException as e:
        print(f"Error sending serial command: {e}")

def normalize_landmarks(landmarks):
    """
    Normalizes hand landmarks to be scale and position invariant.
    Subtracts wrist coordinates and scales by a characteristic distance (e.g., wrist to middle finger MCP).
    """
    if not landmarks:
        return None

    # Convert landmarks to a NumPy array for easier manipulation
    # Each landmark has x, y, z coordinates. We'll use x, y for 2D comparison.
    landmark_coords = np.array([[lm.x, lm.y] for lm in landmarks.landmark])

    # Get wrist landmark (index 0)
    wrist = landmark_coords[0]

    # Get middle finger MCP landmark (index 9) for scaling
    middle_finger_mcp = landmark_coords[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

    # Calculate characteristic distance for scaling
    # Avoid division by zero if hands are too close or not detected properly
    scale_factor = np.linalg.norm(middle_finger_mcp - wrist)
    if scale_factor == 0:
        return None # Cannot normalize if scale factor is zero

    # Normalize: subtract wrist coordinates (translation) and divide by scale factor
    normalized_coords = (landmark_coords - wrist) / scale_factor

    # Flatten the array for comparison
    return normalized_coords.flatten()

def compare_landmarks(ref_landmarks, user_landmarks):
    """
    Compares two sets of normalized landmarks using Euclidean distance.
    Lower distance means higher similarity.
    """
    if ref_landmarks is None or user_landmarks is None:
        return float('inf') # Return a very large number if either is missing

    # Ensure both arrays have the same shape
    if ref_landmarks.shape != user_landmarks.shape:
        return float('inf')

    # Calculate Euclidean distance
    distance = np.linalg.norm(ref_landmarks - user_landmarks)
    return distance

def load_video_references(video_path, signs_config, hands_processor):
    """
    Loads reference hand landmarks and frames for each sign from the video.
    Includes a search window around the timestamp to find a valid hand detection.
    If no hand is found, it uses a black frame and None for landmarks.
    """
    reference_data = {}
    cap_ref_video = cv2.VideoCapture(video_path)

    if not cap_ref_video.isOpened():
        print(f"Error: Could not open reference video file {video_path}.")
        return {}

    print("\n--- Loading Reference Signs from Video ---")
    for sign_label, timestamp_sec in signs_config.items():
        # Seek to the start of the search window
        start_search_time_ms = max(0, (timestamp_sec - REFERENCE_LOAD_SEARCH_WINDOW_SEC / 2) * 1000)
        cap_ref_video.set(cv2.CAP_PROP_POS_MSEC, start_search_time_ms)
        
        hand_found_in_window = False
        search_window_end_time_ms = (timestamp_sec + REFERENCE_LOAD_SEARCH_WINDOW_SEC / 2) * 1000
        
        # Prepare a black frame as a fallback display if no hand is found
        fallback_display_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(fallback_display_frame, f"No Ref for {sign_label}", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
        # Try to find a hand within the search window
        while cap_ref_video.get(cv2.CAP_PROP_POS_MSEC) < search_window_end_time_ms:
            ret, frame = cap_ref_video.read()
            if not ret:
                # End of video reached within search window
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands_processor.process(frame_rgb)

            if results.multi_hand_landmarks:
                normalized_lm = normalize_landmarks(results.multi_hand_landmarks[0])
                if normalized_lm is not None:
                    # Draw landmarks on the reference frame for display
                    display_frame = frame.copy()
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            display_frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())

                    reference_data[sign_label] = {
                        'landmarks': normalized_lm,
                        'frame': display_frame # Store the frame with landmarks drawn for display
                    }
                    print(f"Loaded reference for '{sign_label}' at {cap_ref_video.get(cv2.CAP_PROP_POS_MSEC)/1000:.2f}s (found in window).")
                    hand_found_in_window = True
                    break # Found a hand, move to next sign
            
        if not hand_found_in_window:
            # If no hand found in the window, use the fallback frame and None for landmarks
            reference_data[sign_label] = {
                'landmarks': None,
                'frame': fallback_display_frame
            }
            print(f"Warning: No hand detected in video for '{sign_label}' within search window around {timestamp_sec}s. Using fallback.")

    cap_ref_video.release()
    return reference_data

# --- Main Program ---
def main():
    # Load all reference signs from the video at startup
    reference_sign_templates = load_video_references(VIDEO_PATH, REFERENCE_SIGNS_CONFIG, hands_video)
    
    if not reference_sign_templates:
        print("Error: No reference signs loaded. Please check VIDEO_PATH and REFERENCE_SIGNS_CONFIG timestamps.")
        return

    # Prepare quiz sequence based on loaded signs
    # Sort quiz sequence by timestamp to ensure correct order of quizzes
    quiz_sequence_sorted_items = sorted(REFERENCE_SIGNS_CONFIG.items(), key=lambda item: item[1])
    quiz_sequence = [item[0] for item in quiz_sequence_sorted_items]

    if not quiz_sequence:
        print("No signs available for quiz. Exiting.")
        return

    cap_video = cv2.VideoCapture(VIDEO_PATH)
    cap_webcam = cv2.VideoCapture(0) # 0 for default webcam. Try 1, 2, etc. if 0 fails.

    if not cap_video.isOpened():
        print(f"Error: Could not open video file at {VIDEO_PATH}. Please ensure the video is downloaded and the path is correct.")
        return
    if not cap_webcam.isOpened():
        print("Error: Could not open webcam. Please check if another application is using it or if permissions are granted.")
        print("You might also try changing 'cv2.VideoCapture(0)' to 'cv2.VideoCapture(1)' or 'cv2.VideoCapture(2)'.")
        return

    # Set video to the starting learning time (for continuous playback)
    cap_video.set(cv2.CAP_PROP_POS_MSEC, START_TIME_SEC * 1000)

    last_quiz_completion_time = time.time() # Tracks when the last quiz ended (matched or timed out)
    quiz_active = False
    current_quiz_sign_index = 0 # Start with the first sign in the sorted sequence
    current_quiz_sign_label = quiz_sequence[current_quiz_sign_index]
    
    print("\n--- ASL Learning System Started ---")
    print(f"Learning segment: {START_TIME_SEC}s to {END_TIME_SEC}s (adjusted for new video)")
    print(f"Quizzes will trigger based on video timestamps, with a {QUIZ_INTERVAL_SEC}s delay after each quiz.")
    print("Press 'q' to quit.")

    while cap_webcam.isOpened():
        ret_cam, frame_cam = cap_webcam.read()
        if not ret_cam:
            print("Failed to grab webcam frame. Attempting to re-initialize webcam...")
            cap_webcam.release() # Release the camera
            time.sleep(1) # Wait a bit
            cap_webcam = cv2.VideoCapture(0) # Try re-opening (you might need to try other indices here too)
            if not cap_webcam.isOpened():
                print("Failed to re-initialize webcam. Exiting.")
                break # If re-initialization fails, break the loop
            else:
                print("Webcam re-initialized successfully.")
                continue # Skip to next iteration after re-init

        # Process webcam frame for user's hand
        frame_cam_rgb = cv2.cvtColor(frame_cam, cv2.COLOR_BGR2RGB)
        results_cam = hands_webcam.process(frame_cam_rgb)
        user_landmarks_flat = None
        if results_cam.multi_hand_landmarks:
            user_landmarks_flat = normalize_landmarks(results_cam.multi_hand_landmarks[0])

            # Draw landmarks on the user's webcam feed
            for hand_landmarks in results_cam.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame_cam,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        # --- Video Playback and Quiz Triggering Logic ---
        # Read a new frame from the video for continuous playback
        ret_vid, frame_vid = cap_video.read()
        if not ret_vid:
            # Loop back to start_time_sec if video ends during continuous playback
            cap_video.set(cv2.CAP_PROP_POS_MSEC, START_TIME_SEC * 1000)
            print("Looping video segment.")
            # Reset quiz sequence if video loops
            current_quiz_sign_index = 0
            current_quiz_sign_label = quiz_sequence[current_quiz_sign_index]
            last_quiz_completion_time = time.time() # Reset delay timer
            continue
        
        current_video_time_sec = cap_video.get(cv2.CAP_PROP_POS_MSEC) / 1000
        display_frame_vid = frame_vid # Default: show live video frame

        if not quiz_active:
            # Check if enough time has passed since the last quiz completion AND
            # if the video has reached or passed the timestamp for the next quiz sign.
            if (time.time() - last_quiz_completion_time >= QUIZ_INTERVAL_SEC) and \
               (current_quiz_sign_index < len(quiz_sequence)): # Ensure we haven't run out of signs
                
                next_sign_label = quiz_sequence[current_quiz_sign_index]
                next_sign_timestamp = REFERENCE_SIGNS_CONFIG[next_sign_label]

                if current_video_time_sec >= next_sign_timestamp:
                    # Trigger new quiz
                    current_quiz_sign_label = next_sign_label
                    
                    # Ensure the current quiz sign has a loaded template
                    # If reference_sign_templates[current_quiz_sign_label]['landmarks'] is None, it means no hand was detected during loading
                    if reference_sign_templates[current_quiz_sign_label]['landmarks'] is None:
                        print(f"Warning: No valid reference landmarks for '{current_quiz_sign_label}' were loaded. Skipping direct comparison for this sign.")
                        # In this case, the quiz will still proceed, but the comparison will always fail
                        # unless the user's hand is also not detected, leading to a "No hand detected!" message.
                        # This ensures the quiz sequence continues.

                    quiz_active = True
                    quiz_start_time = time.time() # Start timer for user's response
                    
                    print(f"\n--- QUIZ TIME: Match '{current_quiz_sign_label}' ---")
                    send_serial_command('I', f"QUIZ: {current_quiz_sign_label}") # 'I' for Info/Instruction
                    
                    # When quiz is active, display the static reference frame for the current quiz sign
                    display_frame_vid = reference_sign_templates[current_quiz_sign_label]['frame']
        else:
            # If quiz is active, display the static reference frame for the current quiz sign
            if current_quiz_sign_label and current_quiz_sign_label in reference_sign_templates:
                display_frame_vid = reference_sign_templates[current_quiz_sign_label]['frame']
            else:
                # Fallback if reference frame is somehow missing (shouldn't happen if loaded correctly)
                display_frame_vid = np.zeros((480, 640, 3), dtype=np.uint8) # Black frame
                cv2.putText(display_frame_vid, "LOADING REF...", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


            # --- Quiz Active Logic ---
            # Display current match threshold on the webcam feed
            cv2.putText(frame_cam, f"Threshold: {MATCH_THRESHOLD}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame_cam, f"Match: {current_quiz_sign_label}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


            if current_quiz_sign_label and current_quiz_sign_label in reference_sign_templates:
                reference_landmarks_flat = reference_sign_templates[current_quiz_sign_label]['landmarks']
                
                if reference_landmarks_flat is not None and user_landmarks_flat is not None:
                    similarity = compare_landmarks(reference_landmarks_flat, user_landmarks_flat)
                    cv2.putText(frame_cam, f"Similarity: {similarity:.4f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                    if similarity < MATCH_THRESHOLD:
                        send_serial_command('G', "CORRECT")
                        print(f"✅ Correct! Similarity: {similarity:.4f}")
                        cv2.putText(frame_cam, "CORRECT!", (frame_cam.shape[1] // 2 - 100, frame_cam.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5, cv2.LINE_AA)
                        quiz_active = False # End quiz
                        current_quiz_sign_index += 1 # Move to next sign
                        last_quiz_completion_time = time.time() # Reset timer for next quiz
                        time.sleep(0.5) # Short pause to show success
                    else:
                        send_serial_command('R', "INCORRECT")
                        cv2.putText(frame_cam, "INCORRECT!", (frame_cam.shape[1] // 2 - 120, frame_cam.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5, cv2.LINE_AA)
                else:
                    # Case where reference landmarks are None (no hand detected in video for this sign)
                    # or user's hand is not detected.
                    send_serial_command('R', "NO HAND DETECT") # More generic message
                    cv2.putText(frame_cam, "NO HAND DETECT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            else: # This else block handles if current_quiz_sign_label or reference_sign_templates is invalid
                send_serial_command('R', "REF ERROR")
                cv2.putText(frame_cam, "REF ERROR!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)


            # Check for quiz timeout
            if time.time() - quiz_start_time > QUIZ_DURATION_SEC:
                if quiz_active: # Only if it hasn't been marked correct yet
                    send_serial_command('R', "TIMED OUT")
                    print("❌ Quiz timed out.")
                    cv2.putText(frame_cam, "TIMED OUT!", (frame_cam.shape[1] // 2 - 120, frame_cam.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 165, 255), 5, cv2.LINE_AA) # Orange color
                quiz_active = False # End quiz due to timeout
                current_quiz_sign_index += 1 # Move to next sign
                last_quiz_completion_time = time.time() # Reset timer for next quiz

        # --- Display Frames ---
        # Resize frames to have consistent height for side-by-side display
        h_cam, w_cam, _ = frame_cam.shape
        h_vid, w_vid, _ = display_frame_vid.shape

        # Maintain aspect ratio while resizing video frame to match webcam height
        aspect_ratio_vid = w_vid / h_vid
        display_frame_vid_resized = cv2.resize(display_frame_vid, (int(h_cam * aspect_ratio_vid), h_cam))

        combined_frame = np.hstack((display_frame_vid_resized, frame_cam))
        cv2.imshow('ASL Learning System - Video (Left) | Your Hand (Right)', combined_frame)

        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # --- Cleanup ---
    cap_video.release()
    cap_webcam.release()
    ser.close()
    print("\n--- ASL Learning System Ended ---")

if __name__ == "__main__":
    main()
