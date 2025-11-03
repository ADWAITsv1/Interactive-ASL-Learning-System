import cv2
import mediapipe as mp
import numpy as np
import serial
import time
import collections # For deque to manage quiz sequence

# --- Configuration ---
# Path to your downloaded ASL tutorial video
VIDEO_PATH = '/Users/ady/Desktop/My_projects/Takfujisenseiproj/asl_tutorial.mp4'
# Serial port for your Arduino. IMPORTANT: Change this to your Arduino's port!
SERIAL_PORT = '/dev/cu.usbmodem201912341'
BAUD_RATE = 9600

# This now acts as a delay AFTER a quiz is matched/timed out, before the NEXT sign is considered.
QUIZ_INTERVAL_SEC = 5 # Increased to 5 seconds for a slower pace between quizzes

# Maximum time (in seconds) for the user to match each letter of the sign
QUIZ_DURATION_SEC = 8 # Increased to 8 seconds to give more time for signing

# Similarity threshold for a "correct" match.
# IMPORTANT: This is Euclidean distance. Lower value = closer match.
MATCH_THRESHOLD = 1.5 # You might need to adjust this value based on your testing.

# Duration (in seconds) to search for a hand around the specified timestamp during reference loading
REFERENCE_LOAD_SEARCH_WINDOW_SEC = 2.0 # Increased search window to 2.0 seconds

# --- Reference Signs Configuration (Your Knowledge Base for Letters) ---
# Define the signs (letters) to be quizzed and their exact timestamp (in seconds) in the video.
# These timestamps are based on the image you provided.
REFERENCE_SIGNS_CONFIG = {
    'A': 15.0, 'B': 19.0, 'C': 26.0, 'D': 33.0, 'E': 39.0, 'F': 46.0,
    'G': 51.0, 'H': 56.0, 'I': 62.0, 'J': 69.0, 'K': 80.0, 'L': 86.0,
    'M': 93.0, 'N': 97.0, 'O': 103.0, 'P': 110.0, 'Q': 116.0, 'R': 122.0,
    'S': 128.0, 'T': 133.0, 'U': 139.0, 'V': 145.0, 'W': 151.0, 'X': 155.0,
    'Y': 161.0, 'Z': 171.0
}

# --- Quiz Words Configuration ---
# Define the words to be quizzed. Each word is a list of its constituent letters.
# Ensure all letters in these words exist in REFERENCE_SIGNS_CONFIG.
QUIZ_WORDS = [
    ['D', 'O', 'G'],
    ['C', 'A', 'T'],
    ['H', 'E', 'L', 'L', 'O'],
    ['A', 'S', 'L'],
    ['B', 'O', 'O', 'K'],
    ['F', 'I', 'S', 'H'],
    ['W', 'A', 'T', 'E', 'R'],
    ['H', 'O', 'M', 'E'],
    ['J', 'U', 'M', 'P'],
    ['L', 'O', 'V', 'E']
]

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
    # Load all reference signs (letters) from the video at startup
    reference_letter_templates = load_video_references(VIDEO_PATH, REFERENCE_SIGNS_CONFIG, hands_video)
    
    if not reference_letter_templates:
        print("Error: No reference letter signs loaded. Please check VIDEO_PATH and REFERENCE_SIGNS_CONFIG timestamps.")
        return

    # Prepare quiz words sequence
    # Shuffle the words for a varied quiz experience
    quiz_words_shuffled = list(QUIZ_WORDS)
    np.random.shuffle(quiz_words_shuffled)
    
    # Use a deque to easily cycle through words
    quiz_word_queue = collections.deque(quiz_words_shuffled)

    cap_video = cv2.VideoCapture(VIDEO_PATH) # For continuous background video
    cap_webcam = cv2.VideoCapture(0) # 0 for default webcam. Try 1, 2, etc. if 0 fails.

    if not cap_video.isOpened():
        print(f"Error: Could not open video file at {VIDEO_PATH}. Please ensure the video is downloaded and the path is correct.")
        return
    if not cap_webcam.isOpened():
        print("Error: Could not open webcam. Please check if another application is using it or if permissions are granted.")
        print("You might also try changing 'cv2.VideoCapture(0)' to 'cv2.VideoCapture(1)' or 'cv2.VideoCapture(2)'.")
        return

    # Set background video to start time
    cap_video.set(cv2.CAP_PROP_POS_MSEC, 0) # Start from beginning for background flow

    quiz_active = False
    current_word = None
    current_letter_index = 0
    last_quiz_completion_time = time.time() # Tracks when the last letter quiz ended
    score = 0
    
    print("\n--- ASL Word Quiz System Started ---")
    print("Press 'q' to quit.")
    print("Get ready to sign words!")

    while cap_webcam.isOpened():
        ret_cam, frame_cam = cap_webcam.read()
        if not ret_cam:
            print("Failed to grab webcam frame. Attempting to re-initialize webcam...")
            cap_webcam.release()
            time.sleep(1)
            cap_webcam = cv2.VideoCapture(0)
            if not cap_webcam.isOpened():
                print("Failed to re-initialize webcam. Exiting.")
                break
            else:
                print("Webcam re-initialized successfully.")
                continue

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

        # --- Background Video Playback ---
        ret_vid, frame_vid = cap_video.read()
        if not ret_vid:
            # Loop background video
            cap_video.set(cv2.CAP_PROP_POS_MSEC, 0)
            continue
        
        # --- Quiz Logic ---
        if not quiz_active:
            # Check if enough time has passed since the last letter quiz completion
            if time.time() - last_quiz_completion_time >= QUIZ_INTERVAL_SEC:
                if not current_word: # If no word is active, get a new one
                    if not quiz_word_queue:
                        print("\n--- All words quizzed! Shuffling and repeating. ---")
                        quiz_word_queue = collections.deque(quiz_words_shuffled) # Reshuffle and repeat
                    current_word = quiz_word_queue.popleft()
                    current_letter_index = 0
                    print(f"\n--- NEW WORD: {''.join(current_word)} ---")
                    send_serial_command('I', f"WORD: {''.join(current_word)}")

                # Get the current letter to quiz
                current_letter_label = current_word[current_letter_index]
                
                # Check if this letter has a reference template
                if current_letter_label not in reference_letter_templates or \
                   reference_letter_templates[current_letter_label]['landmarks'] is None:
                    print(f"Warning: No valid reference for '{current_letter_label}'. Skipping this letter.")
                    current_letter_index += 1
                    if current_letter_index >= len(current_word):
                        current_word = None # Move to next word
                    last_quiz_completion_time = time.time()
                    continue

                quiz_active = True
                quiz_start_time = time.time()
                print(f"--- QUIZ: Sign '{current_letter_label}' for word {''.join(current_word)} ---")
                send_serial_command('I', f"SIGN: {current_letter_label}")
        
        # --- Display Elements ---
        # 1. Background Video (full size, then scaled down for corner)
        # 2. User Webcam (main display)
        # 3. Reference Sign (small clip in corner)
        # 4. Quiz Word Text
        # 5. Score Text
        # 6. Countdown Timer

        # Scale down background video for corner display
        small_video_height = frame_cam.shape[0] // 3 # Roughly 1/3rd height of webcam feed
        small_video_width = int(frame_vid.shape[1] * (small_video_height / frame_vid.shape[0]))
        scaled_background_video = cv2.resize(frame_vid, (small_video_width, small_video_height))

        # Get the reference sign frame for the current letter
        display_ref_frame = np.zeros((small_video_height, small_video_width, 3), dtype=np.uint8) # Black placeholder
        if quiz_active and current_letter_label in reference_letter_templates:
            ref_frame_original = reference_letter_templates[current_letter_label]['frame']
            # Resize reference frame to fit the small video corner
            display_ref_frame = cv2.resize(ref_frame_original, (small_video_width, small_video_height))
        else:
            # If quiz not active or no reference, show a placeholder
            cv2.putText(display_ref_frame, "Next Sign", (small_video_width//4, small_video_height//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)


        # Create a blank canvas for the combined display
        # We need space for webcam (right), quiz word (left top), score (left middle), and reference video (right bottom)
        # Let's make the main canvas width sum of webcam + some left panel width
        left_panel_width = 300 # Pixels for quiz word and score
        total_width = frame_cam.shape[1] + left_panel_width
        total_height = frame_cam.shape[0]

        combined_display = np.zeros((total_height, total_width, 3), dtype=np.uint8)
        
        # Place webcam on the right
        combined_display[:, left_panel_width:] = frame_cam

        # Place background video in the top right corner of the webcam feed (or a dedicated corner)
        # For simplicity, let's place it in the bottom right of the combined display for now
        # We can adjust this to be "clip in the right corner of the tab" more precisely later.
        combined_display[total_height - small_video_height:, total_width - small_video_width:] = display_ref_frame


        # Add Quiz Word text on the left panel (top)
        word_text = f"Word: {''.join(current_word) if current_word else '---'}"
        current_letter_text = f"Sign: {current_word[current_letter_index] if current_word and quiz_active else '---'}"
        
        cv2.putText(combined_display, word_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(combined_display, current_letter_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)

        # Add Points Table text on the left panel (middle)
        score_text = f"Score: {score}"
        cv2.putText(combined_display, score_text, (10, total_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

        # --- Countdown Timer ---
        if quiz_active:
            remaining_time = max(0, int(QUIZ_DURATION_SEC - (time.time() - quiz_start_time)))
            countdown_text = f"Time: {remaining_time}s"
            cv2.putText(combined_display, countdown_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2, cv2.LINE_AA) # Orange color

        # --- Quiz Active Logic (for user's hand comparison) ---
        if quiz_active:
            current_letter_ref_landmarks = reference_letter_templates[current_letter_label]['landmarks']
            
            if current_letter_ref_landmarks is not None and user_landmarks_flat is not None:
                similarity = compare_landmarks(current_letter_ref_landmarks, user_landmarks_flat)
                
                # Display similarity on webcam feed
                cv2.putText(frame_cam, f"Sim: {similarity:.4f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                if similarity < MATCH_THRESHOLD:
                    send_serial_command('G', "CORRECT")
                    print(f"✅ Correct! Similarity: {similarity:.4f}")
                    cv2.putText(frame_cam, "CORRECT!", (frame_cam.shape[1] // 2 - 100, frame_cam.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5, cv2.LINE_AA)
                    
                    current_letter_index += 1 # Move to next letter in word
                    if current_letter_index >= len(current_word):
                        # Store the completed word before setting current_word to None
                        completed_word_str = ''.join(current_word) 
                        current_word = None # Word completed, move to next word
                        send_serial_command('G', "WORD DONE!")
                        print(f"--- Word '{completed_word_str}' completed! ---")
                        score += 1 # Increment score ONLY on word completion

                    quiz_active = False # End current letter quiz
                    last_quiz_completion_time = time.time() # Reset timer for next letter/word
                    time.sleep(0.5) # Short pause to show success
                else:
                    send_serial_command('R', "INCORRECT")
                    cv2.putText(frame_cam, "INCORRECT!", (frame_cam.shape[1] // 2 - 120, frame_cam.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5, cv2.LINE_AA)
            else:
                send_serial_command('R', "NO HAND DETECT")
                cv2.putText(frame_cam, "NO HAND DETECT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            # Check for quiz timeout (for current letter)
            if time.time() - quiz_start_time > QUIZ_DURATION_SEC:
                if quiz_active: # Only if it hasn't been matched yet
                    send_serial_command('R', "TIMED OUT")
                    print("❌ Quiz timed out for letter.")
                    cv2.putText(frame_cam, "TIMED OUT!", (frame_cam.shape[1] // 2 - 120, frame_cam.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 165, 255), 5, cv2.LINE_AA)
                    
                    current_letter_index += 1 # Move to next letter
                    if current_letter_index >= len(current_word):
                        # Store the failed word before setting current_word to None
                        failed_word_str = ''.join(current_word)
                        current_word = None # Word timed out, move to next word
                        send_serial_command('R', "WORD FAIL!")
                        print(f"--- Word '{failed_word_str}' failed! ---")

                quiz_active = False # End current letter quiz
                last_quiz_completion_time = time.time() # Reset timer for next letter/word


        cv2.imshow('ASL Word Quiz System', combined_display)

        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # --- Cleanup ---
    cap_video.release()
    cap_webcam.release()
    ser.close()
    print("\n--- ASL Word Quiz System Ended ---")

if __name__ == "__main__":
    main()
